import os
import time
import hashlib
import pickle
import copy
import uuid
import tqdm
import numpy as np
import torch
#import dnnlib
from metrics.image import metric_utils
from metrics.image.metric_utils import get_feature_detector_name, get_feature_detector, FeatureStats
from metrics.image.precision_recall import compute_distances

from multiprocessing import cpu_count
import types
from functools import partial

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as TF
from torchvision.transforms.functional import InterpolationMode
from scipy import linalg
from PIL import Image


def get_new_transform(input_transform, interpolation=InterpolationMode.BILINEAR, antialias=False):
    fn_resize = TF.Resize((299, 299), interpolation=interpolation, antialias=antialias)
    fn_totensor = TF.ToTensor()
    if isinstance(input_transform, TF.ToTensor):
        new_transform = TF.Compose([fn_resize, fn_totensor])
        return new_transform
    elif isinstance(input_transform, TF.Compose):
        ind_start, _ = search_to_tensor(input_transform.transforms)
        new_transform = TF.Compose(input_transform.transforms[:ind_start] + [fn_resize, fn_totensor])
    else:
        raise NotImplementedError
    return new_transform

def _type_cast(x):
    if x.dtype == torch.uint8:
        return x.float() / 255.
    else:
        EPS=1e-12
        assert x.min().item() >= 0.-EPS and x.max().item() <= 1.+EPS
        return x

def _data_transforms_tensor(img_batch: torch.Tensor, transform):
    """
    copied and modified from https://github.com/GaParmar/clean-fid/blob/e1bc9c3ce1337ec70b2ae2a2866bcc621ca96a04/cleanfid/fid.py#L215
    """
    assert img_batch.dtype == torch.uint8
    batch_size = img_batch.size(0)
    resized_batch = torch.zeros(batch_size, 3, 299, 299, device=img_batch.device)
    for idx in range(batch_size):
        curr_img = img_batch[idx]
        img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
        img_pil = Image.fromarray(img_np)
        resized_img = transform(img_pil).to(img_batch.device)
        resized_batch[idx] = resized_img
    return resized_batch

def generator_wrapper(generator, transform):
    for batch in generator:
        if isinstance(batch, list):
            yield [transform(batch[0])] + batch[1:]
        elif isinstance(batch, tuple):
            batch = list(batch)
            yield tuple([transform(batch[0])] + batch[1:])
        else:
            yield transform(batch)

def get_interpolation_mode(interpolation='bilinear'):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    else:
        raise NotImplementedError


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_precision_recall(real_features, gen_features, nhood_size=3, row_batch_size=10000, col_batch_size=10000, rank=0, num_gpus=1):
    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if rank == 0 else None)
        kth = torch.cat(kth) if rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if rank == 0 else 'nan')
    return results['precision'], results['recall']


@torch.no_grad()
def compute_feature_stats_for_dataset(
    opts,
    dataset,
    detector_url, detector_kwargs, detector_cache_dir=None,
    rel_lo=0, rel_hi=1,
    batch_size=64,
    data_loader_kwargs=None,
    transform=None,
    resize_mode='tensor', interpolation='bilinear', antialias=False,
    max_items=None,
    **stats_kwargs,
):
    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        #cache_file = dnnlib.make_cache_dir_path('data-metrics', cache_tag + '.pkl')
        if not os.path.exists(opts.cache_path):
            os.makedirs(opts.cache_path, exist_ok=True)
        cache_file = os.path.join(opts.cache_path, 'data-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, cache_dir=detector_cache_dir, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Initialize transform
    total_processed = 0
    new_transform = None
    resize = None
    total = None

    if resize_mode == 'pil':
        #assert input_dtype == 'float'
        total = len(dataset)
        original_transform = dataset.transform
        new_transform = get_new_transform(original_transform, interpolation=get_interpolation_mode(interpolation), antialias=antialias)
        dataset.transform = new_transform
    else:
        #assert input_dtype == 'float'
        total = len(dataset)
        resize = partial(F.interpolate, size=(299, 299), mode=interpolation, align_corners=False, antialias=antialias)

    # Main loop.
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in tqdm.tqdm(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs),
                                     disable=False if opts.rank == 0 else True,
                                     ncols=80):
        if transform is not None:
            images = transform(images)
        if resize is not None:
            images = resize(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        images = (images*255).clamp(0, 255).to(torch.uint8)
        features = detector(images.to(opts.device), **detector_kwargs) # detector will resize image internally
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats


@torch.no_grad()
def compute_feature_stats_for_generator(
    opts,
    create_generator,
    detector_url, detector_kwargs, detector_cache_dir=None,
    rel_lo=0, rel_hi=1,
    batch_size=64,
    transform=None,
    resize_mode='tensor', interpolation='bilinear', antialias=False,
    max_items=50000,
    **stats_kwargs,
):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- create_generator : Generator
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- max_samples : Setting this value will stop activation when max_samples is reached
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(G_kwargs=opts.G_kwargs,
                    detector_url=detector_url,
                    detector_kwargs=detector_kwargs,
                    stats_kwargs=stats_kwargs,
                   )
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        #cache_file = dnnlib.make_cache_dir_path('gen-metrics', cache_tag + '.pkl')
        if not os.path.exists(opts.cache_path):
            os.makedirs(opts.cache_path, exist_ok=True)
        cache_file = os.path.join(opts.cache_path, cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    stats = FeatureStats(max_items=max_items, **stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, cache_dir=detector_cache_dir, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Create generator.
    max_items_per_gpu = int(np.ceil(max_items / opts.num_gpus))
    dl = create_generator(batch_size=batch_size, num_samples=max_items_per_gpu)

    # Initialize transforms
    total_processed = 0
    new_transform = None
    total = None
    if resize_mode == 'pil':
        # assert antialias == True
        if isinstance(dl, DataLoader):
            #assert input_dtype == 'float'
            total = len(dl)
            original_transform = dl.dataset.transform
            new_transform = get_new_transform(original_transform, interpolation=get_interpolation_mode(interpolation), antialias=antialias)
            dl.dataset.transform = new_transform
        elif isinstance(dl, types.GeneratorType):
            #assert input_dtype == 'uint8'
            total = int(np.ceil(stats.max_items / batch_size)) if stats.max_items is not None else None
            _transform = partial(
                    _data_transforms_tensor,
                    transform=TF.Compose([TF.Resize((299, 299), interpolation=get_interpolation_mode(interpolation), antialias=antialias), TF.ToTensor()]))
            dl = generator_wrapper(dl, transform=_transform)
        else:
            raise NotImplementedError
    elif resize_mode == 'tensor':
        # assert antialias == False
        if isinstance(dl, DataLoader):
            #assert input_dtype == 'float'
            total = len(dl)
            resize = partial(F.interpolate, size=(299, 299), mode=interpolation, align_corners=False, antialias=antialias)
        elif isinstance(dl, types.GeneratorType):
            total = int(np.ceil(stats.max_items / batch_size)) if stats.max_items is not None else None
            resize = TF.Compose([_type_cast, partial(F.interpolate, size=(299, 299), mode=interpolation, align_corners=False, antialias=antialias)])
        else:
            raise NotImplementedError
        dl = generator_wrapper(dl, transform=resize)
    else:
        raise ValueError

    for batch in tqdm.tqdm(dl, total=int(np.ceil(max_items_per_gpu / batch_size)), disable=False if opts.rank == 0 else True, ncols=80):
        # break
        if stats.is_full():
            break

        # ignore labels
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]

        batch = batch.to(opts.device)

        if transform is not None:
            batch = transform(batch)

        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)

        batch = (batch*255).clamp(0, 255).to(torch.uint8)

        # forward
        features = detector(batch, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    if resize_mode == 'pil' and isinstance(dl, DataLoader):
        dl.dataset.transform = original_transform

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats
