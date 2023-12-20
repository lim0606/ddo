import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from lib.models.layers import PointwiseOp


class SpectralConv1d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 modes_height:int,
                 fft_norm='forward',
                 bias=True,
                 skip=False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.half_modes_height = modes_height//2+1
        self.fft_norm = fft_norm

        self._weight = Parameter(torch.view_as_real(torch.complex(
                torch.zeros(out_channels, in_channels, self.half_modes_height),
                torch.zeros(out_channels, in_channels, self.half_modes_height),
            )))
        if bias:
            self.bias = Parameter(torch.empty(1, out_channels, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if skip:
            self.skip = PointwiseOp(in_channels, out_channels, dim=1, scale_factor=1, bias=False)
        else:
            self.skip = None

    @property
    def weight(self):
        return torch.view_as_complex(self._weight)

    def reset_parameters(self) -> None:
        self._weight.data.normal_(std=np.sqrt(2/self.modes_height))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size, in_channels, height = x.shape
        assert in_channels == self.in_channels

        # fft
        xh = torch.fft.rfft(x, dim=-1, norm=self.fft_norm)

        # weight * xh
        out_fft = torch.zeros([batch_size, self.out_channels, height//2+1], device=xh.device, dtype=xh.dtype)
        out_fft[:, :, :self.half_modes_height] = torch.einsum("bix,oix->box",
                                                              xh[:, :, :self.half_modes_height],
                                                              self.weight,
                                                             )

        # ifft
        out = torch.fft.irfft(out_fft, dim=-1, norm=self.fft_norm)

        # bias
        if self.bias is not None:
            out = out + self.bias

        # skip
        if self.skip is not None:
            out = out + self.skip(x)

        return out

    def extra_repr(self) -> str:
        s = [f'{self.in_channels}',
             f'{self.out_channels}',
             f'modes_height={self.modes_height}',
             f'fft_norm={self.fft_norm}',
             ]
        if self.bias is None:
            s += ['bias=False']
        return ', '.join(s)


class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 modes_height:int,
                 modes_width:int,
                 fft_norm='forward', #None,
                 bias=True,
                 skip=False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        assert modes_height == modes_width, 'current version only supports modes_height == modes_width'
        self.half_modes_width = modes_width//2+1
        self.fft_norm = fft_norm

        self._weight = Parameter(torch.view_as_real(torch.complex(
                torch.zeros(out_channels, in_channels, modes_height, self.half_modes_width),
                torch.zeros(out_channels, in_channels, modes_height, self.half_modes_width),
            )))
        if bias:
            self.bias = Parameter(torch.empty(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if skip:
            self.skip = PointwiseOp(in_channels, out_channels, dim=2, scale_factor=1, bias=False)
        else:
            self.skip = None

    @property
    def weight(self):
        return torch.view_as_complex(self._weight)

    def reset_parameters(self) -> None:
        self._weight.data.normal_(std=np.sqrt(2/(self.modes_height*self.modes_width)))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        assert in_channels == self.in_channels

        # fft
        xh = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)

        # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros([batch_size, self.out_channels, height, width//2+1], device=xh.device, dtype=xh.dtype)

        # upper block (truncate high freq)
        out_fft[:, :, :self.modes_height//2, :self.half_modes_width] = torch.einsum("bixy,oixy->boxy",
                                                                                    xh[:, :, :self.modes_height//2, :self.half_modes_width],
                                                                                    self.weight[:, :, :self.modes_height//2],
                                                                                    )
        # Lower block
        out_fft[:, :, -self.modes_height//2:, :self.half_modes_width] = torch.einsum("bixy,oixy->boxy",
                                                                                     xh[:, :, -self.modes_height//2:, :self.half_modes_width],
                                                                                     self.weight[:, :, self.modes_height//2:],
                                                                                     )

        # ifft
        out = torch.fft.irfft2(out_fft, dim=(-2, -1), norm=self.fft_norm)

        # bias
        if self.bias is not None:
            out = out + self.bias

        # skip
        if self.skip is not None:
            out = out + self.skip(x)

        return out

    def extra_repr(self) -> str:
        s = [f'{self.in_channels}',
             f'{self.out_channels}',
             f'modes_height={self.modes_height}',
             f'modes_width={self.modes_width}',
             f'fft_norm={self.fft_norm}',
             ]
        if self.bias is None:
            s += ['bias=False']
        return ', '.join(s)


def group_norm(x, ref_freq, num_groups, dim=1, EPS=1e-6):
    if ref_freq is None:
        ref_freq = x
    if dim == 1:
        batch_size, channels, height1 = x.shape
        batch_size2, channels2, height2 = ref_freq.shape
        assert batch_size == batch_size2 and channels == channels2
        x = x.reshape(batch_size*num_groups, -1, height1)
        ref_freq = ref_freq.reshape(batch_size*num_groups, -1, height2)
        norm = torch.clip(torch.linalg.vector_norm(ref_freq, dim=(1,2), keepdim=True), min=EPS)
        x = x / norm
        return x.reshape(batch_size, channels, height1)
    elif dim == 2:
        batch_size, channels, height1, width1 = x.shape
        batch_size2, channels2, height2, width2 = ref_freq.shape
        assert batch_size == batch_size2 and channels == channels2
        x = x.reshape(batch_size*num_groups, -1, height1, width1)
        ref_freq = ref_freq.reshape(batch_size*num_groups, -1, height2, width2)
        norm = torch.clip(torch.linalg.vector_norm(ref_freq, dim=(1,2,3), keepdim=True), min=EPS)
        x = x / norm
        return x.reshape(batch_size, channels, height1, width1)
    else:
        raise NotImplementedError


class SpectralGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, modes_height, affine=True, eps=1e-6, cutoff=False, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        n_modes = (modes_height,)
        self.order = len((n_modes))
        self.num_groups = num_groups
        self.fft_norm = 'forward'
        self.eps = eps
        self.cutoff = cutoff

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        half_modes = [m//2 for m in n_modes]
        self.half_modes = half_modes

        self.half_modes_height = self.half_modes[0]

        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.ones(1, num_channels))
            self.bias = Parameter(torch.zeros(1, num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward_1d(self, x):
        batchsize, channels, height = x.shape

        # fft
        xh = torch.fft.rfft(x, dim=-1, norm=self.fft_norm)

        ref_fft = torch.zeros([batchsize, self.num_channels,  height//2 + 1], device=xh.device, dtype=xh.dtype)
        ref_fft[:, :, :self.half_modes_height+1] = xh[:, :, :self.half_modes_height+1]

        # normalize
        if self.cutoff:
            ref = torch.fft.irfft(ref_fft, dim=-1, norm=self.fft_norm)
            out = group_norm(ref, ref_freq=ref_fft, num_groups=self.num_groups, dim=1, EPS=self.eps)
        else:
            out = group_norm(x, ref_freq=ref_fft, num_groups=self.num_groups, dim=1, EPS=self.eps)

        # return
        if self.affine:
            return self.weight[...,None] * out + self.bias[...,None]
        else:
            return out

    def forward_2d(self, x):
        if not hasattr(self, 'half_modes_width'):
            self.half_modes_width = self.half_modes_height
        batch_size, channels, height, width = x.shape

        # fft
        xh = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)

        # The output will be of size (batch_size, self.num_channels, x.size(-2), x.size(-1)//2 + 1)
        ref_fft = torch.zeros([batch_size, self.num_channels, height, width//2 + 1], dtype=xh.dtype, device=xh.device)
        # upper block (truncate high freq)
        ref_fft[:, :, :self.half_modes_height,  :self.half_modes_width+1] = xh[:, :, :self.half_modes_height, :self.half_modes_width+1]
        # lower block
        ref_fft[:, :, -self.half_modes_height:, :self.half_modes_width+1] = xh[:, :, -self.half_modes_height:, :self.half_modes_width+1]

        # normalize
        if self.cutoff:
            ref = torch.fft.irfft2(ref_fft, dim=(-2, -1), norm=self.fft_norm)
            out = group_norm(ref, ref_freq=ref_fft, num_groups=self.num_groups, dim=2, EPS=self.eps)
        else:
            out = group_norm(x, ref_freq=ref_fft, num_groups=self.num_groups, dim=2, EPS=self.eps)

        # return
        if self.affine:
            return self.weight[...,None,None] * out + self.bias[...,None,None]
        else:
            return out

    def forward(self, x):
        dim = x.dim()
        if dim == 3:
            return self.forward_1d(x)
        elif dim == 4:
            return self.forward_2d(x)
        else:
            raise NotImplementedError

    def extra_repr(self) -> str:
        return (f'(num_groups): {self.num_groups}\n'
                f'(num_channels): {self.num_channels}\n'
                f'(modes_height): {self.n_modes[0]}\n'
                f'(affine): {self.affine}\n'
                f'(cutoff): {self.cutoff}'
                )
