import os
import functools
import time
import argparse
import json
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process

from utils import utils
from utils import datasets
from utils.ema import EMA
from utils.utils import save_checkpoint, load_checkpoint
from utils.visualize import get_grid_image
from utils.evaluate import compute_feature_stats_for_dataset, compute_feature_stats_for_generator
from utils.evaluate import calculate_frechet_distance, calculate_precision_recall
from metrics.image import metric_utils

from lib.diffusion import diffuse
from lib.diffusion import LinearDiffusion, BlurringDiffusion, DenoisingDiffusion
from lib.models.unet import UNet2d
from lib.models.fourier_unet import FNOUNet2d
from lib.models.aliasfree_dct import Upsample2d, Downsample2d


def low_discrepancy_rand(batch_size, device=None):
    u0 = torch.rand(1).item()
    t2 = ( u0 + torch.linspace(0, batch_size-1, batch_size, device=device) / batch_size ) % 1
    return t2

def low_discrepancy_randint(batch_size, k, device=None):
    u0 = torch.rand(1).item()
    t2 = torch.floor( ( u0 + torch.linspace(0, batch_size-1, batch_size, device=device) / batch_size ) * k % k ).long()
    return t2

def get_meshgrid_2d(height, batch_size=1):
    x = torch.linspace(0, height-1, height) / height
    return torch.concat((
        x[None,None,None].repeat(batch_size, 1, height, 1),
        x[None,None,:,None].repeat(batch_size, 1, 1, height)), dim=1)

def get_act(act:str=None):
    if act is None:
        return nn.SiLU()
    if act == "gelu":
        return nn.GELU()
    elif act == "silu":
        return nn.SiLU()
    else:
        raise NotImplementedError


def xv_to_image(x, v=None):
    batch_size, num_points, _ = x.shape
    img_height = int(np.sqrt(num_points))
    assert img_height*img_height == num_points
    if v is not None:
        return (x.reshape(batch_size, img_height, img_height, -1).permute(0, 3, 1, 2).contiguous(),
                v.reshape(batch_size, img_height, img_height, -1).permute(0, 3, 1, 2).contiguous())
    else:
        return (x.reshape(batch_size, img_height, img_height, -1).permute(0, 3, 1, 2).contiguous(),
                None)

def x_to_image(x):
    batch_size, num_points, _ = x.shape
    img_height = int(np.sqrt(num_points))
    assert img_height*img_height == num_points
    return x.reshape(batch_size, img_height, img_height, -1).permute(0, 3, 1, 2).contiguous()

def image_to_xv(x, v=None):
    if v is not None:
        return (x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]),
                v.permute(0, 2, 3, 1).reshape(v.shape[0], -1, v.shape[1]))

    else:
        return (x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]),
                None)

def get_mgrid(dim, img_height):
    grid = torch.linspace(0, img_height-1, img_height) / img_height
    if dim == 1:
        grid = grid[None, None]
    elif dim == 2:
        grid = torch.cat([grid[None,None,...,None].repeat(1, 1, 1, img_height),
                          grid[None,None,None].repeat(1, 1, img_height, 1)], dim=1)
    elif dim == 3:
        grid = torch.cat([grid[None,None,...,None,None].repeat(1, 1, 1, img_height, img_height),
                          grid[None,None,None,...,None].repeat(1, 1, img_height, 1, img_height),
                          grid[None,None,None,None].repeat(1, 1, img_height, img_height, 1)], dim=1)
    else:
        raise NotImplementedError
    return grid

def identity(x):
    return x

def to_center(x):
    return x * 2. - 1.

def to_01(x):
    return (x + 1.) / 2.

def to_01_clip(x):
    x = (x + 1.) / 2.
    return torch.clip(x, 0, 1)

def logit(x, alpha=0.05):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    return y

def inverse_logit(y, alpha=0.05):
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    return x

def from_sdf_to_01(x):
    return (-1.*x + 1.) / 2.

def from_sdf_to_01_clip(x):
    x = (-1.*x + 1.) / 2.
    return torch.clip(x, 0, 1)

def from_sdf_to_mask(x):
    return (x < 0.).float()

def x_to_mask(x):
    x = x_to_image(x)
    return from_sdf_to_mask(x)


@torch.no_grad()
def sample_image(gen_sde, batch_size, img_height, num_steps,
                 lmbd=0.,
                 s_min=1e-4,
                 transform=None,
                 sampler='denoise',
                 upsample=False,
                 upsample_resolution=64,
                 filter_size=9,
                 use_radial=False,
                 disable_tqdm=False,
                 **kwargs,
                ):
    # init
    device = gen_sde.device
    v = get_mgrid(2, img_height).repeat(batch_size, 1, 1, 1).to(gen_sde.device)

    # upsample
    scale_factor = None
    if upsample and v.shape[-1] < upsample_resolution:
        input_height = v.shape[-1]
        scale_factor = int(upsample_resolution/input_height)
        assert scale_factor*input_height == upsample_resolution

        upsample = Upsample2d(
            scale_factor=scale_factor,
            filter_size=filter_size,
            use_radial=use_radial,
        ).to(device)
        downsample = Downsample2d(
            scale_factor=1/scale_factor,
            filter_size=filter_size,
            use_radial=use_radial,
        ).to(device)

        v = upsample(v)

    # init
    x_0 = gen_sde.gp.sample(v.shape[0], v.shape[-1], ch=gen_sde.in_channels)

    # sample and plot
    x_T = diffuse(gen_sde,
                  num_steps=num_steps,
                  x_0=x_0,
                  v=v,
                  lmbd=lmbd,
                  s_min=s_min,
                  sampler=sampler,
                  disable_tqdm=disable_tqdm,
                  keep_all_samples=False,
                 )[-1].to(device) # sample

    # downsample
    if upsample and scale_factor is not None:
        x_T = downsample(x_T)

    # post process
    if transform is not None:
        x_T = transform(x_T)

    return x_T

def create_generator(num_samples, batch_size, gen_sde, img_height, num_steps,
                     lmbd=0., s_min=1e-4, sampler='denoise', transform=None,
                     upsample=False,
                     upsample_resolution=64,
                     filter_size=9,
                     use_radial=False,
                    ):
    num_iters = int(np.ceil(num_samples / batch_size))
    for i in range(num_iters):
        x_T = sample_image(
                gen_sde=gen_sde,
                batch_size=batch_size,
                img_height=img_height,
                num_steps=num_steps,
                lmbd=lmbd,
                s_min=s_min,
                sampler=sampler,
                transform=transform,
                upsample=upsample, upsample_resolution=upsample_resolution, filter_size=filter_size, use_radial=use_radial,
                disable_tqdm=True,
                )
        yield x_T.float()


class Model(nn.Module):
    def __init__(self, model:str, *args, **kwargs):
        super().__init__()
        if model == 'unet2d':
            model_class = UNet2d
        elif model == 'fnounet2d':
            model_class = FNOUNet2d
        else:
            raise NotImplementedError
        self.model = model_class(*args, **kwargs)

    @property
    def in_channels(self):
        return self.model.in_channels

    def forward(self, x, temp, v, **kwargs):
        return self.model.forward(x=x, temp=temp, v=v, **kwargs)

def get_model(args):
    # init forward sde
    gp_config = argparse.Namespace()
    gp_config.device = 'cuda'
    gp_config.exponent = args.gp_exponent if hasattr(args, 'gp_exponent') else None
    gp_config.length_scale = args.gp_length_scale if hasattr(args, 'gp_length_scale') else None
    gp_config.sigma = args.gp_sigma if hasattr(args, 'gp_sigma') else None
    gp_config.modes_height = args.gp_modes if hasattr(args, 'gp_modes') else None
    gp_config.modes_width = args.gp_modes if hasattr(args, 'gp_modes') else None
    gp_config.modes = args.gp_modes if hasattr(args, 'gp_modes') else None
    gp_config.window = args.gp_window if hasattr(args, 'gp_window') else None

    disp_config = argparse.Namespace()
    disp_config.sigma_blur_min = args.sigma_blur_min
    disp_config.sigma_blur_max = args.sigma_blur_max

    if args.disp_method is None:
        inf_sde = LinearDiffusion(
            dim=args.coord_dim,
            ch=args.input_dim,
            ns_method=args.ns_method,
            gp_type=args.gp_type,
            gp_config=gp_config,
        )
    else:
        inf_sde = BlurringDiffusion(
            dim=args.coord_dim,
            ch=args.input_dim,
            ns_method=args.ns_method,
            disp_method=args.disp_method,
            disp_config=disp_config,
            gp_type=args.gp_type,
            gp_config=gp_config,
        )

    # init model
    drift_q = Model(
        model=args.model,

        modes_height=args.modes if hasattr(args, 'modes') else None,
        modes_width=args.modes if hasattr(args, 'modes') else None,

        in_channels=args.input_dim,
        in_height=args.train_img_height,
        ch=args.ch,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        act=get_act(args.act) if hasattr(args, 'act') else nn.SiLU(),
        use_pos=args.use_pos if hasattr(args, 'use_pos') else False,

        use_radial=args.use_radial if hasattr(args, 'use_radial') else False,

        resamp_with_conv=not args.discard_resamp_with_conv if hasattr(args, 'discard_resamp_with_conv') else True,
        use_pointwise_op=args.use_pointwise_op if hasattr(args, 'use_pointwise_op') else False,

        norm=args.norm if hasattr(args, 'norm') else  None,
    )

    # init model
    gen_sde = DenoisingDiffusion(inf_sde, model=drift_q, timestep_sampler=args.timestep_sampler, use_clip=args.use_clip, weight_method=args.weight_method)

    return gen_sde

def init_model(args, resume:bool=None):
    # init model
    gen_sde = get_model(args).cuda()

    # broadcast params
    utils.broadcast_params(gen_sde.parameters(), args.distributed)

    # init optimizer
    if args.optimizer == "adam":
        gen_sde_optimizer = torch.optim.Adam(gen_sde.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        gen_sde_optimizer = torch.optim.AdamW(gen_sde.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:
        raise ValueError

    # add ema
    gen_sde_optimizer = EMA(gen_sde_optimizer, ema_decay=args.ema_decay)

    # init scheduler
    if args.lr_rampup_kimg > 0:
        def warmup(current_step: int):
            return min(current_step * args.train_batch_size / max(args.lr_rampup_kimg * 1000, 1e-8), 1)
        gen_sde_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_sde_optimizer, lr_lambda=warmup)
    else:
        gen_sde_scheduler = None

    # resume
    checkpoint_file = os.path.join(args.folder_path, args.checkpoint_file)
    resume = args.resume if resume is None else resume
    if resume and os.path.exists(checkpoint_file):
        print('loading the model.')
        (gen_sde,
         gen_sde_optimizer,
         gen_sde_scheduler,
         count,
         best_fid_score,
         ) = load_checkpoint(checkpoint_file, gen_sde, gen_sde_optimizer, gen_sde_scheduler)#, state_update_rule=custom_rule)
        print('loaded the model at iter {:d}'.format(count))
    else:
        count = 0
        best_fid_score = 1e10

    return gen_sde, gen_sde_optimizer, gen_sde_scheduler, count, best_fid_score

def save_model(args, global_step, gen_sde, best_fid_score, checkpoint_name='checkpoint_fid.pt'):
    if args.global_rank == 0:
        # Because we already swapped EMA parameters, we only save the gen_sde models here.
        content = {'global_step': global_step,
                   'args': args,
                   'best_fid_score': best_fid_score,
                   'gen_sde_state_dict': gen_sde.state_dict(),
                   }
        checkpoint_file = os.path.join(args.folder_path, checkpoint_name)
        torch.save(content, checkpoint_file)
    return

def load_model(checkpoint_file, distributed=None, state_update_rule=None):
    # load model
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    count = checkpoint['global_step']
    args = checkpoint['args']
    gen_sde = get_model(args).cuda()
    if distributed is None:
        distributed = args.distributed
    utils.broadcast_params(gen_sde.parameters(), distributed)
    if state_update_rule is not None:
        gen_sde.load_state_dict(state_update_rule(checkpoint['gen_sde_state_dict']))
    else:
        gen_sde.load_state_dict(checkpoint['gen_sde_state_dict'])
    best_fid_score = checkpoint['best_fid_score']
    print('loaded the model at iter {:d}'.format(count))
    return gen_sde, count, best_fid_score, args


def evaluate(args, gen_sde, img_height, count, eval_metric='fid', upsample=False, upsample_resolution=64, filter_size=9, use_radial=False):
    # get datasets
    if args.dataset.startswith('mnistsdf_'):
        dataset_name = 'mnistsdf_{:d}'.format(img_height)
    else:
        dataset_name = args.dataset
    dataset, _, _ = datasets.get_datasets(
        dataset=dataset_name,
        root=args.data,
        centered=False,
    )
    dataset.name = dataset_name

    if eval_metric == 'fid':
        detector = 'inception'
    elif eval_metric == 'pr':
        detector = 'vgg'
    else:
        raise ValueError

    # init detector
    if detector == 'inception':
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    elif detector == 'vgg':
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    else:
        raise NotImplementedError
    detector_kwargs = dict(return_features=True)

    # get data activations
    real_opts = metric_utils.MetricOptions(
        num_gpus=args.global_size,
        rank=args.global_rank,
        cache=True,
        cache_path=args.fid_dir,
    )
    real_stats = compute_feature_stats_for_dataset(
        real_opts,
        dataset,
        detector_url,
        detector_kwargs,
        detector_cache_dir=args.fid_dir,
        batch_size=args.eval_batch_size_per_gpu,
        transform=x_to_mask if args.dataset.startswith('mnistsdf_') else None,
        resize_mode=args.eval_resize_mode,
        interpolation=args.eval_interpolation,
        antialias=args.eval_antialias,
        max_items=50000,
        capture_all=True,
    )
    real_features = real_stats.get_all_torch().to(real_opts.device)
    torch.distributed.barrier()

    # get model activations
    gen_opts = metric_utils.MetricOptions(
        num_gpus=args.global_size,
        rank=args.global_rank,
        cache=args.eval_cache if hasattr(args, 'eval_cache') else False,
        cache_path=args.folder_path,
        #verbose=True,
        G_kwargs=dict(
           count=count,
           num_steps=args.num_steps,
           img_height=img_height,
           sampler=args.sampler,
           lmdb=args.eval_lmbd,
        ),
    )
    fn_create_generator = functools.partial(
            create_generator,
            gen_sde=gen_sde,
            batch_size=args.eval_batch_size_per_gpu,
            img_height=img_height,
            sampler=args.sampler,
            num_steps=args.num_steps,
            lmbd=args.eval_lmbd,
            transform=from_sdf_to_mask if args.dataset.startswith('mnistsdf_') else args.reverse,
            upsample=upsample,
            upsample_resolution=upsample_resolution,
            filter_size=filter_size,
            use_radial=use_radial,
    )
    gen_stats = compute_feature_stats_for_generator(
        gen_opts,
        fn_create_generator,
        detector_url,
        detector_kwargs,
        detector_cache_dir=args.fid_dir,
        batch_size=args.eval_batch_size_per_gpu,
        resize_mode=args.eval_resize_mode,
        interpolation=args.eval_interpolation,
        antialias=args.eval_antialias,
        max_items=args.eval_num_samples,
        capture_all=True,
    )
    gen_features = gen_stats.get_all_torch().to(gen_opts.device)
    torch.distributed.barrier()

    # compute fid
    if args.global_rank == 0 and eval_metric == 'fid':
        real_features_ = real_features.cpu().numpy()
        m0 = np.mean(real_features_, axis=0)
        s0 = np.cov(real_features_, rowvar=False)

        gen_features_ = gen_features.cpu().numpy()
        m = np.mean(gen_features_, axis=0)
        s = np.cov(gen_features_, rowvar=False)

        fid = calculate_frechet_distance(m0, s0, m, s)
    else:
        fid = -1.
    torch.distributed.barrier()

    # compute precision recall
    if args.global_rank == 0 and eval_metric == 'pr':
        precision, recall = calculate_precision_recall(
            real_features,
            gen_features,
            nhood_size=args.nhood_size if hasattr(args, 'nhood_size') else 3,
            row_batch_size=10000,
            col_batch_size=10000,
            rank=args.global_rank,
            num_gpus=args.global_size,
        )
    else:
        precision, recall = -1., -1.
    torch.distributed.barrier()

    return fid, precision, recall

def train(args):
    logging, writer = utils.common_init(args.global_rank, args.seed, args.folder_path)
    logging.info(f'folder path: {args.folder_path}')
    logging.info(str(args))

    # dataset
    train_loader, _, _ = datasets.get_loaders_eval(
        dataset=args.dataset,
        root=args.data,
        distributed=args.distributed,
        batch_size=args.train_batch_size_per_gpu,
        centered=False,
    )
    num_iters_per_epoch = len(train_loader)
    args.epochs = int(np.ceil(args.num_iterations / num_iters_per_epoch))

    # init model
    (gen_sde,
     gen_sde_optimizer,
     gen_sde_scheduler,
     count,
     best_fid_score,
     ) = init_model(args)
    logging.info(gen_sde)

    # upsample
    if args.upsample and args.train_img_height < args.upsample_resolution:
        scale_factor = int(args.upsample_resolution/args.train_img_height)
        assert scale_factor*args.train_img_height == args.upsample_resolution
        upsample = Upsample2d(
            scale_factor=scale_factor,
            filter_size=args.upsample_filter_size,
            use_radial=args.use_radial,
        ).cuda()

    # transform
    if args.transform == "center":
        args.forward = forward = to_center
        args.reverse = reverse = to_01_clip
    elif args.transform == "sdf":
        args.forward = forward = x_to_image
        args.reverse = reverse = from_sdf_to_01_clip
    elif args.transform == "logit":
        args.forward = forward = logit
        args.reverse = reverse = inverse_logit
    else:
        args.forward = forward = identity
        args.reverse = reverse = identity

    # init
    x_plot, _ = next(iter(train_loader))
    if args.dequantize:
       x_plot = x_plot * 255 / 256 + torch.rand_like(x_plot) / 256
    x_plot = args.forward(x_plot).cuda()

    # plot data
    if count == 0:
        x_plot = args.reverse(x_plot)
        nrow = int(args.vis_batch_size**0.5)
        writer.add_image('data/samples', get_grid_image(x_plot[:nrow**2].cpu(), nrow=nrow, pad_value=0, padding=2, to_numpy=False), count)
        writer.flush()
        logging.info('Visualized data')

    # train
    start_time = time.time()
    epoch = count // num_iters_per_epoch
    if count >= args.num_iterations:
        not_finished = False
        logging.info('Finished training at iter {:d}'.format(count))
    else:
        not_finished = True
    while not_finished:
        if args.distributed:
            train_loader.sampler.set_epoch(count + args.seed)
        for (x, _) in train_loader:
            # init model and cdae
            gen_sde.train()

            # init grad
            gen_sde_optimizer.zero_grad()

            # init batch
            if args.dequantize:
                x = x * 255 / 256 + torch.rand_like(x) / 256
            x = args.forward(x).cuda()

            # upsample
            if args.upsample:
                x = upsample(x)

            # init v
            v = get_mgrid(2, x.shape[-1]).repeat(x.shape[0], 1, 1, 1).cuda()

            # dsm
            loss = gen_sde.dsm(x, v).mean() # forward and compute loss

            # backward
            loss.backward()

            # broadcast
            utils.average_gradients(gen_sde.parameters(), args.distributed)

            # clip graidient norm
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(gen_sde.parameters(), args.clip_grad_norm)

            # update
            gen_sde_optimizer.step()
            count += 1
            epoch = count // num_iters_per_epoch
            if gen_sde_scheduler is not None:
                gen_sde_scheduler.step()

            ''' print '''
            if args.global_rank == 0 and count == 1 or count % args.print_every == 0:
                elapsed = time.time() - start_time # set log info
                logging.info('| epoch {:6d}  | iter {:6d}  | {:5.2f} ms/step  | loss {:5.3f}  | lr {:f}'
                      .format(
                      epoch,
                      count,
                      elapsed*1000/args.print_every,
                      loss.item(),
                      gen_sde_scheduler.get_last_lr()[-1] if gen_sde_scheduler is not None else args.lr,
                      ))
                writer.add_scalar('train/loss/iter', loss.item(), count)
                writer.add_scalar('train/lr/iter', gen_sde_scheduler.get_last_lr()[-1] if gen_sde_scheduler is not None else args.lr, count)
                start_time = time.time() # reset log info

            if args.save_every > 0 and count % args.save_every == 0:
                save_checkpoint(args, count, best_fid_score, gen_sde, gen_sde_optimizer, checkpoint_name='checkpoint.pt')
                logging.info('Saved checkpoint at iter {:d}'.format(count))

            if args.ckpt_every > 0 and count % args.ckpt_every == 0:
                save_checkpoint(args, count, best_fid_score, gen_sde, gen_sde_optimizer, checkpoint_name='checkpoint_{}.pt'.format(count))
                logging.info('Saved checkpoint at iter {:d}'.format(count))

            if args.vis_every > 0 and count % args.vis_every == 0:
                # switch to EMA parameters
                if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
                    gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                if args.global_rank == 0:
                    sample = sample_image(gen_sde, batch_size=args.vis_batch_size, img_height=32, sampler=args.sampler, num_steps=args.num_steps, lmbd=args.eval_lmbd, transform=args.reverse, clip=False)
                    nrow = int(args.vis_batch_size**0.5)
                    writer.add_image(f'train/samples/32/{args.sampler}-{args.num_steps}', get_grid_image(sample[:nrow**2].cpu(), nrow=nrow, pad_value=0, padding=2, to_numpy=False), count)
                    writer.flush()

                    sample = sample_image(gen_sde, batch_size=args.vis_batch_size, img_height=64, sampler=args.sampler, num_steps=args.num_steps, lmbd=args.eval_lmbd, transform=args.reverse, clip=False)
                    nrow = int(args.vis_batch_size**0.5)
                    writer.add_image(f'train/samples/64/{args.sampler}-{args.num_steps}', get_grid_image(sample[:nrow**2].cpu(), nrow=nrow, pad_value=0, padding=2, to_numpy=False), count)
                    writer.flush()

                    sample = sample_image(gen_sde, batch_size=args.vis_batch_size, img_height=128, sampler=args.sampler, num_steps=args.num_steps, lmbd=args.eval_lmbd, transform=args.reverse, clip=False)
                    nrow = int(args.vis_batch_size**0.5)
                    writer.add_image(f'train/samples/128/{args.sampler}-{args.num_steps}', get_grid_image(sample[:nrow**2].cpu(), nrow=nrow, pad_value=0, padding=2, to_numpy=False), count)
                    writer.flush()

                    logging.info('Visualized samples at iter {:d}'.format(count))

                    if args.plot:
                        matplotlib.use(MYBACKEND)
                        sample = sample.permute(0, 2, 3, 1).reshape(sample.shape[0], -1, sample.shape[1])
                        plot_contourf(sample[:min(args.vis_batch_size, 4)].cpu(),
                                      v[:min(args.vis_batch_size, 4)].cpu(),
                                      img_height=args.eval_img_height,
                                      nrows=1, ncols=4, figsize=(20,4))

                # switch back to original parameters
                if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
                    gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            if args.eval_every > 0 and count % args.eval_every == 0:
                # switch to EMA parameters
                if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
                    gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                eval_start_time = time.time()
                val_fid_ema, _, _ = evaluate(args, gen_sde=gen_sde, img_height=args.eval_img_height, count=count, eval_metric='fid')
                eval_elapsed = time.time() - eval_start_time
                logging.info('-'*60)
                logging.info('| iter {:6d} | {:5.2f} sec | fid {:5.3f}  ({:} steps)'
                      .format(
                      count,
                      eval_elapsed,
                      val_fid_ema,
                      args.num_steps,
                      ))
                logging.info('-'*60)
                writer.add_scalar('train/fid{}/{:d}x{:d}/{:d}/orig/iter'.format(
                                      '_ema' if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema') else '',
                                      args.eval_img_height, args.eval_img_height,
                                      args.num_steps),
                                  val_fid_ema,
                                  count,
                                 )
                writer.flush()

                # save best fid model
                if args.global_rank == 0 and val_fid_ema < best_fid_score:
                    best_fid_score = val_fid_ema
                    logging.info('Saving the model for FID.')
                    save_model(args, count, gen_sde, best_fid_score, checkpoint_name='checkpoint_fid.pt')

                # switch back to original parameters
                if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
                    gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            if count >= args.num_iterations:
                save_checkpoint(args, count, best_fid_score, gen_sde, gen_sde_optimizer, checkpoint_name='checkpoint.pt')
                not_finished = False
                logging.info('Finished training at iter {:d}'.format(count))
                break

    return gen_sde, gen_sde_optimizer, gen_sde_scheduler, count, None


def test(args):
    logging, writer = utils.common_init(args.global_rank, args.seed, args.folder_path, suffix='test')
    logging.info(f'folder path: {args.folder_path}')
    logging.info(str(args))

    # load model
    if args.model is None:
        checkpoint_file = args.checkpoint_file
    else:
        checkpoint_file = os.path.join(args.folder_path, args.checkpoint_file)
    if os.path.basename(checkpoint_file) in {'checkpoint_init.pt', 'checkpoint_fid.pt'}:
        gen_sde, count, best_fid_score, train_args = load_model(checkpoint_file)
        train_args.num_proc_node = args.num_proc_node
        train_args.num_process_per_node = args.num_process_per_node
        train_args.node_rank = args.node_rank
        train_args.local_rank = args.local_rank
        train_args.global_rank = args.global_rank
        train_args.master_address = args.master_address
        train_args.master_port = args.master_port

        train_args.global_size = args.global_size
        train_args.eval_batch_size_per_gpu = args.eval_batch_size_per_gpu
        train_args.eval_batch_size_per_gpu = args.eval_batch_size

        train_args.eval_img_height = args.eval_img_height
        train_args.eval_fid = args.eval_fid
        train_args.eval_pr = args.eval_pr
        train_args.eval_num_samples = args.eval_num_samples
        train_args.eval_resize_mode = args.eval_resize_mode
        train_args.eval_interpolation = args.eval_interpolation
        train_args.eval_antialias = args.eval_antialias
        train_args.eval_cache = args.eval_cache
        train_args.fid_dir = args.fid_dir

        train_args.num_steps = args.num_steps
        train_args.sampler = args.sampler
        train_args.s_min = args.s_min
        train_args.eval_lmbd = args.eval_lmbd

        args = train_args
        gen_sde_optimizer = None
        args.eval_use_ema = False
    else:
        gen_sde, gen_sde_optimizer, _, count, best_fid_score = init_model(args, resume=True)

    # transform
    if args.transform == "center":
        args.forward = forward = to_center
        args.reverse = reverse = to_01_clip
    elif args.transform == "sdf":
        args.forward = forward = x_to_image
        args.reverse = reverse = from_sdf_to_01_clip
    elif args.transform == "logit":
        args.forward = forward = logit
        args.reverse = reverse = inverse_logit
    else:
        args.forward = forward = identity
        args.reverse = reverse = identity

    # switch to EMA parameters
    if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
        gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    # evaluate
    fid, precision, recall = -1., -1., -1.
    if args.eval_fid:
        eval_start_time = time.time()
        fid, _, _ = evaluate(args, gen_sde=gen_sde, img_height=args.eval_img_height, count=count, eval_metric='fid')
        eval_elapsed = time.time() - eval_start_time
        logging.info('-'*60)
        logging.info('| iter {:6d} | {:5.2f} sec | ({:d} steps)'.format(count, eval_elapsed, args.num_steps))
        logging.info('| fid {:5.3f}'.format(fid))
        logging.info('-'*60)
        writer.add_scalar('test/fid{}/{:d}x{:d}/{:d}/iter'.format(
                              '_ema' if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema') else '',
                              args.eval_img_height, args.eval_img_height,
                              args.num_steps),
                          fid,
                          count,
                         )
    if args.eval_pr:
        eval_start_time = time.time()
        _, precision, recall = evaluate(args, gen_sde=gen_sde, img_height=args.eval_img_height, count=count, eval_metric='pr')
        eval_elapsed = time.time() - eval_start_time
        logging.info('-'*60)
        logging.info('| iter {:6d} | {:5.2f} sec | ({:d} steps)'.format(count, eval_elapsed, args.num_steps))
        logging.info('| prec {:5.3f}'.format(precision))
        logging.info('| rec {:5.3f}'.format(recall))
        logging.info('-'*60)
        writer.add_scalar('test/prec{}/{:d}x{:d}/{:d}/iter'.format(
                              '_ema' if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema') else '',
                              args.eval_img_height, args.eval_img_height,
                              args.num_steps),
                          precision,
                          count,
                         )
        writer.add_scalar('test/recall{}/{:d}x{:d}/{:d}/iter'.format(
                          '_ema' if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema') else '',
                          args.eval_img_height, args.eval_img_height,
                          args.num_steps),
                          recall,
                          count,
                         )
    writer.close()

    # switch back to original parameters
    if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
        gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    return fid, precision, recall

def save_model_after_load_checkpoint(args):
    logging, writer = utils.common_init(args.global_rank, args.seed, args.folder_path, suffix='save')
    logging.info(f'folder path: {args.folder_path}')
    logging.info(str(args))

    # init model
    (gen_sde,
     gen_sde_optimizer,
     gen_sde_scheduler,
     count,
     best_fid_score,
     ) = init_model(args)
    logging.info(gen_sde)

    # switch to EMA parameters
    if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
        gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    # save best fid model
    if args.global_rank == 0:
        save_model(args, count, gen_sde, best_fid_score, checkpoint_name=f'checkpoint_iter{count}.pt')
    logging.info(f'folder_path: {args.folder_path}')
    logging.info(f'saved the model at iter {count}')

    # switch back to original parameters
    if args.eval_use_ema and hasattr(gen_sde_optimizer, 'swap_parameters_with_ema'):
        gen_sde_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    return


_folder_name_key_tuples = [
    ('model', ''),
    ('use_pos', 'v'),
    ('modes', 'md'),
    ('act', ''),
    ('ch', 'ch'),
    ('ch_mult', 'm'),
    ('num_res_blocks', 'nr'),
    ('use_radial', 'rf'),
    ('discard_resamp_with_conv', 'drsc'),
    ('use_pointwise_op', 'pw'),
    ('norm', ''),

    ('timestep_sampler', 't'),
    ('ns_method', 'n'),
    # ('ns_logsnr_min', 'nmn'),
    # ('ns_logsnr_max', 'nmx'),
    # ('ns_beta_min', 'bmn'),
    # ('ns_beta_max', 'bmx'),

    ('disp_method', 'd'),
    # ('min_scale', 'mn'),
    ('sigma_blur_min', 'smn'),
    ('sigma_blur_max', 'smx'),

    ('gp_type', 'gp'),
    ('gp_exponent', 'p'),
    ('gp_length_scale', 'ls'),
    ('gp_sigma', 'gs'),
    ('gp_modes', 'gm'),
    ('gp_window', 'gw'),

    ('train_img_height', 'nht'),

    ('upsample', 'up'),
    ('upsample_resolution', 'res'),

    ('transform', 'tr'),

    ('weight_method', 'w'),
    ('train_batch_size', 'b'),

    ('optimizer', ''),
    ('lr', 'lr'),
    ('lr_rampup_kimg', 'lrru'),
    ('clip_grad_norm', 'cg'),
    ('weight_decay', 'wd'),
    ('ema_decay', 'ema'),
    ('seed', 'sd'),
]

def get_args(*args, **kwargs):
    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--command_type", type=str, default='train', choices=['train', 'test', 'save'])

    # i/o
    parser.add_argument('--exp_path', type=str, default='results')
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--ckpt_every', type=int, default=100000)
    parser.add_argument('--eval_every', type=int, default=50000)
    parser.add_argument('--vis_every', type=int, default=10000)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)

    # dataset
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data', type=str, default='datasets', help='data root')

    # optimization
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--vis_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay factor')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 of Adam optimizer')
    parser.add_argument('--num_iterations', type=int, default=2000000,
                        help='number of iterations for training')
    parser.add_argument('--lr_rampup_kimg', type=int, default=0)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--weight_method', type=str, default=None)

    # train
    parser.add_argument('--train_img_height', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--coord_dim', type=int, default=2)
    # parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--upsample', action='store_true', default=False)
    parser.add_argument('--upsample_resolution', type=int, default=None)
    parser.add_argument('--upsample_filter_size', type=int, default=9)

    # model
    parser.add_argument("--dequantize", action="store_true", default=False)
    parser.add_argument("--transform", type=str, choices=["center", "sdf", "logit"], default=None)
    parser.add_argument("--model", type=str, default=None)#"unet2d")
    parser.add_argument("--modes", type=int, default=None)
    # parser.add_argument("--radius", type=float, default=None)
    # parser.add_argument("--mid_channels", type=int, default=None)
    parser.add_argument("--act", type=str, default=None, choices=["silu", "gelu"])
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", type=eval, default=None, #"(1,2,4,8)",
                        help="channel multiplier")
    parser.add_argument("--num_res_blocks", type=int, default=1,
                        help="num of residual blocks per each resolution")
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--norm", type=str, default=None, choices=["group_norm", "identity"])
    parser.add_argument("--use_radial", action='store_true', default=False)
    parser.add_argument("--discard_resamp_with_conv", action='store_true', default=False)
    parser.add_argument("--use_pointwise_op", action='store_true', default=False)
    parser.add_argument("--use_pos", action='store_true', default=True)

    # diffusion
    parser.add_argument("--timestep_sampler", type=str, default="uniform", choices=["uniform", "low_discrepancy"])
    parser.add_argument("--ns_method", type=str, default='vp_cosine')
    parser.add_argument("--disp_method", type=str, default=None)#'sine')
    # parser.add_argument("--min_scale", type=float, default=0.)
    parser.add_argument("--sigma_blur_min", type=float, default=0.)
    parser.add_argument("--sigma_blur_max", type=float, default=0.)

    # noise
    parser.add_argument("--gp_type", type=str, default="exponential")#, choices=["independent", "exponential", "scp", "rfp"])
    parser.add_argument("--gp_exponent", type=float, default=None)#1.5)
    parser.add_argument("--gp_length_scale", type=float, default=None)#0.1)
    parser.add_argument("--gp_sigma", type=float, default=None)#1.0)
    parser.add_argument("--gp_modes", type=int, default=None)
    parser.add_argument("--gp_window", type=str, default=None)#'triangle_beta_0.6')

    # eval (sampler)
    parser.add_argument('--use_clip', action='store_true', default=False)
    parser.add_argument('--num_steps', type=int, default=500,
                        help='number of integration steps for sampling')
    parser.add_argument('--sampler', type=str, default='denoise', choices=['denoise', 'em'])
    parser.add_argument('--s_min', type=float, default=1e-4)
    parser.add_argument('--eval_lmbd', type=float, default=0.)

    # eval
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pt')
    parser.add_argument('--eval_img_height', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='input batch size for evaluation')
    parser.add_argument('--eval_use_ema', action='store_true', default=False)
    parser.add_argument('--eval_fid', action='store_true', default=False)
    parser.add_argument('--eval_pr', action='store_true', default=False)
    parser.add_argument('--eval_num_samples', type=int, default=5000,
                        help='number of samples to evaluate metrics')
    parser.add_argument('--eval_resize_mode', type=str, default='tensor', choices=['tensor', 'pil'])
    parser.add_argument('--eval_interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'])
    parser.add_argument('--eval_antialias', action='store_true', default=False)
    parser.add_argument('--eval_cache', action='store_true', default=False)
    parser.add_argument('--fid_dir', type=str, default='datasets/fid-stats',
                        help='path to directory where fid related files are stored')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus for each node')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--global_size', type=int, default=1,
                        help='number of process among all the processes')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default=None,
                        help='port for master')

    return parser.parse_args(*args, **kwargs)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == '__main__':
    # get args
    args = get_args()

    # set port
    if args.master_port is None:
        free_port = 6020 + np.random.randint(100)
        while is_port_in_use(free_port):
            free_port += 1
        args.master_port = str(free_port)

    # folder_name
    _folder_name = []
    for k, abbr in _folder_name_key_tuples:
        if not hasattr(args, k) or getattr(args, k) is None:
            continue
        if k.startswith('upsample_') and args.upsample is False:
            continue
        if k == "use_pos" and args.use_pos is False:
            continue
        if k in ['min_scale', 'sigma_blur_min', 'sigma_blur_max'] and args.disp_method is None:
            continue
        if k in ['ch_mult']:
            _folder_name += [abbr+''.join([str(i) for i in getattr(args, k)])]
        elif k == "use_pos" and args.use_pos:
            _folder_name += [abbr]
        else:
            _folder_name += [abbr+str(getattr(args, k)).lower()]
    folder_name = '-'.join(_folder_name)
    args.folder_path = os.path.join(args.exp_path, args.dataset, folder_name)

    if not os.path.isdir(args.folder_path):
        os.makedirs(args.folder_path)
    with open(os.path.join(args.folder_path, 'args.txt'), 'w') as out:
        out.write(json.dumps(args.__dict__, indent=4))

    # init
    args.global_size = global_size = args.num_proc_node * args.num_process_per_node

    # multiprocessing
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        pass

    # init dataset
    _, _, _ = datasets.get_loaders_eval(args.dataset, args.data, False, batch_size=1, centered=args.centered if hasattr(args, 'centered') else False, num_workers=1)

    # init main function
    if args.command_type == 'train':
        main = train
    elif args.command_type == 'test':
        main = test
    elif args.command_type == 'save':
        main = save_model_after_load_checkpoint
    else:
        raise ValueError

    # run
    if global_size > 1:
        args.distributed = True
        processes = []
        for rank in range(args.num_process_per_node):
            args.local_rank = rank
            args.global_rank = global_rank = rank + args.node_rank * args.num_process_per_node
            args.train_batch_size_per_gpu = args.train_batch_size // global_size
            assert args.train_batch_size_per_gpu * global_size == args.train_batch_size
            args.eval_batch_size_per_gpu = args.eval_batch_size // global_size
            assert args.eval_batch_size_per_gpu * global_size == args.eval_batch_size
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = False
        args.train_batch_size_per_gpu = args.train_batch_size
        args.eval_batch_size_per_gpu = args.eval_batch_size
        utils.init_processes(0, global_size, main, args)
