"""
Copied and modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
Copied and modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# noinspection PyProtectedMember
from torch.nn.init import _calculate_fan_in_and_fan_out


def parse_kwargs(prefix, keep_prefix=False, **kwargs):
    new_kwargs = {}
    for key, val in kwargs.items():
        if key.startswith(prefix):
            if keep_prefix:
                new_key = key
            else:
                new_key = key.split(prefix,1)[-1]
            new_kwargs.update({new_key: val})
    return new_kwargs


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def group_norm(out_ch):
    return nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)


def upsample1d(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module('up_nn', nn.Upsample(scale_factor=2, mode='nearest'))
    if with_conv:
        up.add_module('up_conv', conv1d(in_ch, in_ch, kernel_size=3, stride=1))
    return up


def upsample2d(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module('up_nn', nn.Upsample(scale_factor=2, mode='nearest'))
    if with_conv:
        up.add_module('up_conv', conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return up


def downsample1d(in_ch, with_conv):
    if with_conv:
        down = conv1d(in_ch, in_ch, kernel_size=3, stride=2)
    else:
        down = nn.AvgPool1d(2)
    return down


def downsample2d(in_ch, with_conv):
    if with_conv:
        down = conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    else:
        down = nn.AvgPool2d(2, 2)
    return down


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv1d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, padding=1, bias=True, padding_mode='zeros',
            init_scale=1.):
    conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias, padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1, padding=1, bias=True, padding_mode='zeros',
            init_scale=1.):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias, padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


# class ChannelShuffle(nn.Module):
#     def __init__(self, in_channels:int) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.register_buffer('ind', torch.randperm(in_channels))
# 
#     def forward(self, input):
#         return input[:,self.ind]
# 
#     def extra_repr(self):
#         return 'in_channels={}'.format(self.in_channels)
# 
# def stacked_channel_conv2d_(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1, padding=1, bias=True, padding_mode='zeros',
#                             init_scale=1., in_ch_per_group=32):
#     in_ch_per_group = in_ch_per_group if in_ch_per_group is not None else in_planes
#     assert in_planes % in_ch_per_group == 0
#     if kernel_size in [1, (1, 1)]:
#         conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
#                          kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
#                          bias=bias, padding_mode=padding_mode,
#                          )
#         variance_scaling_init_(conv.weight, scale=init_scale)
#         if bias:
#             nn.init.zeros_(conv.bias)
#     else:
#         conv = nn.Sequential()
# 
#         conv1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, groups=in_planes//in_ch_per_group,
#                           kernel_size=(3, 3), stride=1, padding=1, dilation=1,
#                           bias=bias, padding_mode=padding_mode,
#                           )
#         variance_scaling_init_(conv1.weight, scale=1.)
#         if bias:
#             nn.init.zeros_(conv1.bias)
#         conv.add_module('conv1', conv1)
# 
#         channel_shuffle = ChannelShuffle(in_planes)
#         conv.add_module('channel_shuffle', channel_shuffle)
# 
#         conv2 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, groups=in_planes//in_ch_per_group,
#                           kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
#                           bias=bias, padding_mode=padding_mode,
#                           )
#         variance_scaling_init_(conv2.weight, scale=1.)
#         if bias:
#             nn.init.zeros_(conv2.bias)
#         conv.add_module('conv2', conv2)
# 
#         conv3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias)
#         variance_scaling_init_(conv3.weight, scale=init_scale)
#         if bias:
#             nn.init.zeros_(conv3.bias)
#         conv.add_module('conv3', conv3)
#     return conv
# 
# def separable_conv2d_(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1, padding=1, bias=True, padding_mode='zeros',
#                       init_scale=1., in_ch_per_group=32, shuffle_channel=False):
#     in_ch_per_group = in_ch_per_group if in_ch_per_group is not None else in_planes
#     assert in_planes % in_ch_per_group == 0
#     conv = nn.Sequential()
#     if kernel_size in [3, (3, 3)]:
#         if shuffle_channel:
#             channel_shuffle = ChannelShuffle(in_planes)
#             conv.add_module('channel_shuffle', channel_shuffle)
#         channel_conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, groups=in_planes//in_ch_per_group,
#                                kernel_size=(3, 3), stride=stride, padding=padding, dilation=dilation,
#                                bias=bias, padding_mode=padding_mode,
#                                )
#         variance_scaling_init_(channel_conv.weight, scale=1.)
#         if bias:
#             nn.init.zeros_(channel_conv.bias)
#         conv.add_module('channel_conv', channel_conv)
#     else:
#         assert kernel_size in [1, (1, 1)]
#     depth_conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias)
#     variance_scaling_init_(depth_conv.weight, scale=init_scale)
#     if bias:
#         nn.init.zeros_(depth_conv.bias)
#     conv.add_module('depth_conv', depth_conv)
#     return conv


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.sigmoid(x) * x


class SelfAttention1d(nn.Module):
    """
    copied modified from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py#L29
    copied modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66
    """

    def __init__(self, in_channels, normalize=group_norm):
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_k = conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_v = conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, init_scale=0.)
        self.softmax = nn.Softmax(dim=-1)
        if normalize is not None:
            self.norm = normalize(in_channels)
        else:
            self.norm = nn.Identity()

    # noinspection PyUnusedLocal
    def forward(self, x, temb=None):
        """ t is not used """
        #_, C, H, W = x.size()
        xsz = list(x.shape)
        C = xsz[1]
        HW = np.prod(xsz[2:])

        h = self.norm(x)
        q = self.attn_q(h).view(-1, C, HW) #H * W)
        k = self.attn_k(h).view(-1, C, HW) #H * W)
        v = self.attn_v(h).view(-1, C, HW) #H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = self.softmax(attn)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(*xsz) #h.view(-1, C, H, W)
        h = self.proj_out(h)

        assert h.shape == x.shape
        return x + h


class SelfAttention2d(nn.Module):
    """
    copied modified from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py#L29
    copied modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66
    """

    def __init__(self, in_channels, normalize=group_norm):
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_k = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_v = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, init_scale=0.)
        self.softmax = nn.Softmax(dim=-1)
        if normalize is not None:
            self.norm = normalize(in_channels)
        else:
            self.norm = nn.Identity()

    # noinspection PyUnusedLocal
    def forward(self, x, temb=None):
        """ t is not used """
        #_, C, H, W = x.size()
        xsz = list(x.shape)
        C = xsz[1]
        HW = np.prod(xsz[2:])

        h = self.norm(x)
        q = self.attn_q(h).view(-1, C, HW) #H * W)
        k = self.attn_k(h).view(-1, C, HW) #H * W)
        v = self.attn_v(h).view(-1, C, HW) #H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = self.softmax(attn)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(*xsz) #h.view(-1, C, H, W)
        h = self.proj_out(h)

        assert h.shape == x.shape
        return x + h


def get_sinusoidal_positional_embedding(timesteps: torch.Tensor,
                                        embedding_dim: int,
                                        scale: float = 10000.,
                                        ):
    """
    Copied and modified from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90
    From Fairseq in https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    #assert len(timesteps.size()) == 1  # and timesteps.dtype == tf.int32
    assert len(timesteps.size()) == 1 or len(timesteps.size()) == 2  # and timesteps.dtype == tf.int32
    if len(timesteps.size()) == 1:
        batch_size = timesteps.size(0)
        index_dim = 1
    else:
        batch_size, index_dim = timesteps.size()
        timesteps = timesteps.reshape(batch_size*index_dim)
    timesteps = timesteps.to(torch.get_default_dtype())#float()
    device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(scale) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    # emb = torch.arange(num_embeddings, dtype=torch.float, device=device)[:, None] * emb[None, :]
    emb = timesteps[..., None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad to the last dimension
      # emb = torch.cat([emb, torch.zeros(num_embeddings, 1, device=device)], dim=1)
      emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [batch_size*index_dim, embedding_dim]
    return emb.reshape(batch_size, index_dim*embedding_dim)


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, pos_dim=1, act=Swish()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        self.main = nn.Sequential(
            dense(embedding_dim*pos_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class ResidualBlock1d(nn.Module):
    def __init__(self,
                 in_ch,
                 temb_ch:int=None,
                 out_ch=None,
                 conv_shortcut=False,
                 dropout=0.,
                 normalize=group_norm,
                 act=Swish(),
                 cond_type="improved",
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.temb_ch = temb_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.act = act
        assert cond_type in ["ddpm", "improved"]
        self.cond_type = cond_type

        if temb_ch is not None and temb_ch > 0:
            self.temb_proj = dense(temb_ch, out_ch if cond_type == "ddpm" else out_ch*2)
        else:
            self.temb_proj = None
        self.norm1 = normalize(in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv1d(in_ch, out_ch)
        self.norm2 = normalize(out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout1d(p=dropout) if dropout > 0. else nn.Identity()
        self.conv2 = conv1d(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = conv1d(in_ch, out_ch)
            else:
                self.shortcut = conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb=None):
        # forward conv1
        h = x
        h = self.act(self.norm1(h))
        h = self.conv1(h)

        # add timestep embedding
        if self.temb_proj is not None:
            if self.cond_type == "ddpm":
                v = self.temb_proj(self.act(temb))[:, :, None]
                h = self.norm2(h + v)
            else:
                w, b = self.temb_proj(self.act(temb))[:, :, None].chunk(2, dim=1)
                h = self.norm2(h) * (w + 1) + b
        else:
            h = self.norm2(h)

        # forward conv2
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # shortcut
        x = self.shortcut(x)

        # combine and return
        assert x.shape == h.shape
        return x + h

    def extra_repr(self) -> str:
        if self.temb_proj is not None:
            s = [f'(cond_type): {self.cond_type}']
        else:
            s = []
        return ', '.join(s)


class ResidualBlock2d(nn.Module):
    def __init__(self,
                 in_ch,
                 temb_ch:int=None,
                 out_ch=None,
                 conv_shortcut=False,
                 dropout=0.,
                 normalize=group_norm,
                 act=Swish(),
                 cond_type="improved",
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.temb_ch = temb_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.act = act
        assert cond_type in ["ddpm", "improved"]
        self.cond_type = cond_type

        if temb_ch is not None and temb_ch > 0:
            self.temb_proj = dense(temb_ch, out_ch if cond_type == "ddpm" else out_ch*2)
        else:
            self.temb_proj = None
        self.norm1 = normalize(in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv2d(in_ch, out_ch)
        self.norm2 = normalize(out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0. else nn.Identity()
        self.conv2 = conv2d(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = conv2d(in_ch, out_ch)
            else:
                self.shortcut = conv2d(in_ch, out_ch, kernel_size=(1, 1), padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb=None):
        # forward conv1
        h = x
        h = self.act(self.norm1(h))
        h = self.conv1(h)

        # add timestep embedding
        if self.temb_proj is not None:
            if self.cond_type == "ddpm":
                v = self.temb_proj(self.act(temb))[:, :, None, None]
                h = self.norm2(h + v)
            else:
                w, b = self.temb_proj(self.act(temb))[:, :, None, None].chunk(2, dim=1)
                h = self.norm2(h) * (w + 1) + b
        else:
            h = self.norm2(h)

        # forward conv2
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # shortcut
        x = self.shortcut(x)

        # combine and return
        assert x.shape == h.shape
        return x + h

    def extra_repr(self) -> str:
        if self.temb_proj is not None:
            s = [f'(cond_type): {self.cond_type}']
        else:
            s = []
        return ', '.join(s)


class UNet(nn.Module):
    def __init__(self,
                 in_dim,
                 in_channels,
                 in_height,
                 ch,
                 out_channels=None,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(),#(16,),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=Swish(),
                 normalize=group_norm,
                 use_time_embedding=True,
                 **kwargs,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.in_height = in_height
        self.ch = ch
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize
        self.use_time_embedding = use_time_embedding

        res_kwargs = parse_kwargs('res_', keep_prefix=False, **kwargs)
        conv_kwargs = parse_kwargs('conv_', keep_prefix=True, **kwargs)
        res_kwargs.update(conv_kwargs)


        if self.in_dim == 1:
            conv_init = conv1d
            ResidualBlock =  ResidualBlock1d
            SelfAttention = SelfAttention1d
            Downsample = downsample1d
            Upsample = upsample1d
        elif self.in_dim == 2:
            conv_init = conv2d
            ResidualBlock =  ResidualBlock2d
            SelfAttention = SelfAttention2d
            Downsample = downsample2d
            Upsample = upsample2d
        else:
            raise NotImplementedError

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = in_height
        in_ch = in_channels
        temb_ch = ch * 4 if self.use_time_embedding else 0
        assert in_ht % 2 ** (num_resolutions - 1) == 0, "in_height doesn't satisfy the condition"

        # Timestep embedding
        if self.use_time_embedding:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
                act=act,
            )

        # Downsampling
        self.begin_conv = conv_init(in_ch, ch)
        unet_chs = [ch]
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize,
                    )
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                block_modules['{}b_downsample'.format(i_level)] = Downsample(out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [
            ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize)]
        mid_modules += [SelfAttention2d(in_ch, normalize=normalize)]
        mid_modules += [
            ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch + unet_chs.pop(),
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                in_ch = out_ch
            # Upsample
            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = Upsample(out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            # convert list of modules to a module list, and append to a list
            up_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # End
        self.end_conv = nn.Sequential(
            normalize(in_ch) if normalize is not None else nn.Identity(),
            self.act,
            conv_init(in_ch, out_channels, init_scale=0.),
        )

    # noinspection PyMethodMayBeStatic
    def _compute_cond_module(self, module, x, temb):
        for m in module:
            x = m(x, temb)
        return x

    # noinspection PyArgumentList,PyShadowingNames
    def forward(self, x, temp, *args, **kwargs):
        # Init
        B, C, H, W = x.size()

        # Timestep embedding
        if self.use_time_embedding:
            temb = self.temb_net(temp)
            assert list(temb.shape) == [B, self.ch * 4]
        else:
            temb = None

        # Downsampling
        hs = [self.begin_conv(x)]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(hs[-1], temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                    h = attn_block(h, temb)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                downsample = block_modules['{}b_downsample'.format(i_level)]
                hs.append(downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            # Residual blocks for this resolution
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                    h = attn_block(h, temb)
            # Upsample
            if i_level != 0:
                upsample = block_modules['{}b_upsample'.format(i_level)]
                h = upsample(h)
        assert not hs

        # End
        h = self.end_conv(h)
        assert list(h.size()) == [x.size(0), self.out_channels, x.size(2), x.size(3)]
        return h


class UNet1d(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, in_dim=1, **kwargs)


class UNet2d(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, in_dim=2, **kwargs)


if __name__ == '__main__':

    # Modified UNet for 32x32 images J. Ho et al., '20, Denoising Diffusion Probabilistic Models
    model = UNet(
            in_channels=3,
            in_height=32,
            ch=128,
            ch_mult=(1, 2, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16,),
            resamp_with_conv=True,
            dropout=0,
            )
    print(model)

    # Modified UNet for 256x256 images J. Ho et al., '20, Denoising Diffusion Probabilistic Models
    model = UNet(
        in_channels=3,
        in_height=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
        dropout=0,
    )
    print(model)

    x_ = torch.randn(1, 3, 256, 256)
    t = torch.zeros(1)
    output = model(x_, t)
    print(x_.size(), output.size())
    assert x_.size() == output.size()
