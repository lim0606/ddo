import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty import torch_dct
from lib.dct import LinearDCT
from torch_utils import misc


#----------------------------------------------------------------------------

def _parse_scaling_1d(scaling):
    if isinstance(scaling, int):
        scaling = [scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sy = scaling[0]
    assert sy >= 1
    return sy

def _parse_padding_1d(padding):
    if isinstance(padding, int):
        padding = [padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 1:
        pady = padding[0]
        padding = [pady, pady]
    pady0, pady1 = padding
    return pady0, pady1

def _get_filter_size_1d(f):
    if f is None:
        return 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1]
    fh = f.shape[-1]
    with misc.suppress_tracer_warnings():
        fh = int(fh)
    misc.assert_shape(f, [fh][:f.ndim])
    assert fh >= 1
    return fh

def _parse_scaling_2d(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding_2d(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size_2d(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------

def design_lowpass_filter_dct(numtaps, cutoff, width, fs, radial=False, EPS=1e-6):
    assert numtaps >= 1
    assert numtaps % 2 == 1

    # Identity filter.
    if numtaps == 1:
        return None

    # Separable Kaiser low-pass filter.
    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        f = torch.as_tensor(f, dtype=torch.float32)
        return f / (f[f.shape[-1]//2:].sum() * 2.)

    # Radially symmetric jinc-based filter.
    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r + EPS)
    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w = np.kaiser(numtaps, beta)
    f *= np.outer(w, w)
    f = torch.as_tensor(f, dtype=torch.float32)
    return f / (f[f.shape[-2]//2:,f.shape[-1]//2:].sum() * 4.)

#----------------------------------------------------------------------------

def dct_conv1d(x, weight, dct=None, idct=None):
    if dct is None:
        dct = functools.partial(torch_dct.dct, norm=None)
    if idct is None:
        idct = functools.partial(torch_dct.idct, norm=None)
    x_freq = dct(x)
    f_freq = dct(weight)
    return idct(f_freq * x_freq)

def dct_conv2d(x, weight, dct=None, idct=None):
    if dct is None:
        dct = functools.partial(torch_dct.dct_2d, norm=None)
    if idct is None:
        idct = functools.partial(torch_dct.idct_2d, norm=None)
    x_freq = dct(x)
    f_freq = dct(weight)
    return idct(f_freq * x_freq)

#----------------------------------------------------------------------------

def upfirdn1d_dct(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1., dct=None, idct=None):
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 3
    if f is None:
        f = torch.ones([1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height = x.shape
    upy = _parse_scaling_1d(up)
    assert upy in [1, 2]
    downy = _parse_scaling_1d(down)
    pady0, pady1 = _parse_padding_1d(padding)
    assert pady0 >= 0 and pady1 >= 0

    # Pad or crop.
    # x = F.pad(x, [pady0, pady1], mode='replicate')
    x = F.pad(x, [pady0, pady1], mode='reflect')

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height + pady0 + pady1, 1])
    x = F.pad(x, [0, upy - 1], mode='constant')
    x = x.reshape([batch_size, num_channels, (in_height + pady0 + pady1) * upy])

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([1, 1] + [1] * f.ndim)
    f_padded = torch.zeros(1, 1, x.shape[-1], dtype=torch.float32, device=x.device)
    f_padded[:,:,:f.shape[-1]//2+1] = f[:,:,f.shape[-1]//2:]
    x = dct_conv1d(x, f_padded, dct=dct, idct=idct)

    # Crop
    x = x.reshape([batch_size, num_channels, in_height + pady0 + pady1, upy])
    x = F.pad(x, [0, 0, -pady0, -pady1])
    x = x.reshape([batch_size, num_channels, in_height * upy])

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy]
    return x

def upfirdn2d_dct(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1., dct=None, idct=None):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling_2d(up)
    assert upy in [1, 2] and upx in [1, 2]
    downx, downy = _parse_scaling_2d(down)
    padx0, padx1, pady0, pady1 = _parse_padding_2d(padding)
    assert padx0 >= 0 and padx1 >= 0 and pady0 >= 0 and pady1 >= 0

    # Pad or crop.
    # x = F.pad(x, [pady0, pady1, padx0, padx1], mode='replicate')
    x = F.pad(x, [pady0, pady1, padx0, padx1], mode='reflect')

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height + pady0 + pady1, 1, in_width + padx0 + padx1, 1])
    x = F.pad(x, [0, upy - 1, 0, 0, 0, upx - 1])
    x = x.reshape([batch_size, num_channels, (in_height + pady0 + pady1) * upy, (in_width + padx0 + padx1) * upx])

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([1, 1] + [1] * f.ndim)
    if f.ndim == 4:
        f_padded = torch.zeros(1, 1, x.shape[-2], x.shape[-1], dtype=torch.float32, device=x.device)
        f_padded[:,:,:f.shape[-2]//2+1,:f.shape[-1]//2+1] = f[:,:,f.shape[-2]//2:,f.shape[-1]//2:]
        x = dct_conv2d(x, f_padded, dct=dct, idct=idct)
    else:
        f1_padded = torch.zeros(1, 1, x.shape[-2], x.shape[-1], dtype=torch.float32, device=x.device)
        f1_padded[:,:,0,:f.shape[-1]//2+1] = f[:,:,f.shape[-1]//2:]
        f2_padded = torch.zeros(1, 1, x.shape[-2], x.shape[-1], dtype=torch.float32, device=x.device)
        f2_padded[:,:,:f.shape[-1]//2+1,0] = f[:,:,f.shape[-1]//2:]
        x_freq = dct(x)
        f1_freq = dct(f1_padded)
        f2_freq = dct(f2_padded)
        x = idct(f1_freq * f2_freq * x_freq) / np.sqrt(np.pi * 5)

    # Crop
    x = x.reshape([batch_size, num_channels, in_height + pady0 + pady1, upy, in_width + padx0 + padx1, upx])
    x = F.pad(x, [0, 0, -padx0, -padx1, 0, 0, -pady0, -pady1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------

class _Upsample1d(nn.Module):
    def __init__(
        self,
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        padding             = None
    ):
        super().__init__()
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        self.register_buffer('up_filter', None, persistent=False)
        self._in_size = None
        self.dct = LinearDCT(dim=1, dct_type='dct', norm=None)
        self.idct = LinearDCT(dim=1, dct_type='idct', norm=None)

    @property
    def device(self):
        up_filter = self.get_buffer('up_filter')
        if up_filter is None:
            return None
        else:
            return up_filter.device

    def initialize(self, in_size, padding=None):
        in_sampling_rate = in_size
        in_cutoff = in_sampling_rate/2
        in_stopband = in_cutoff * self.stopband_rel
        in_half_width = np.maximum(in_stopband, in_sampling_rate/2) - in_cutoff

        out_sampling_rate = out_size = 2*in_size
        out_cutoff = out_sampling_rate/2
        out_stopband = out_cutoff * self.stopband_rel
        out_half_width = np.maximum(out_stopband, out_sampling_rate/2) - out_cutoff

        if self._in_size is not None and self._in_size == in_size:
            return
        else:
            self._initialize(
                in_size=in_size,
                out_size=out_size,
                in_sampling_rate=in_sampling_rate,
                out_sampling_rate=out_sampling_rate,
                in_cutoff=in_cutoff,
                out_cutoff=out_cutoff,
                in_half_width=in_half_width,
                out_half_width=out_half_width,
                padding=padding,
            )

    def _initialize(
        self,
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).
        padding=None,
    ):
        if padding is None:
            padding = self.filter_size // 2
        self._in_size = in_size
        self._out_size = out_size
        self.in_size = np.broadcast_to(np.asarray(in_size), [1])
        self.out_size = np.broadcast_to(np.asarray(out_size), [1])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = self.filter_size * self.up_factor if self.up_factor > 1 else 1
        if self.up_taps % 2 == 0:
            self.up_taps += 1
        self.up_filter = design_lowpass_filter_dct(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate, radial=False,
        ).to(self.device)

        # Compute padding.
        self.padding = padding

    def forward(self, x, **kwargs):
        self.initialize(in_size=x.shape[-1])
        x = upfirdn1d_dct(x=x, f=self.up_filter, up=self.up_factor, padding=self.padding,
                          gain=self.up_factor**2,
                          dct=self.dct, idct=self.idct,
                          ) # Upsample.
        return x

class _Downsample1d(nn.Module):
    def __init__(
        self,
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        padding             = None,
    ):
        super().__init__()
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        self.register_buffer('down_filter', None, persistent=False)
        self._in_size = None
        self.dct = LinearDCT(dim=1, dct_type='dct', norm=None)
        self.idct = LinearDCT(dim=1, dct_type='idct', norm=None)

    @property
    def device(self):
        down_filter = self.get_buffer('down_filter')
        if down_filter is None:
            return None
        else:
            return down_filter.device

    def initialize(self, in_size, padding=None):
        in_sampling_rate = in_size
        in_cutoff = in_sampling_rate/2
        in_stopband = in_cutoff * self.stopband_rel
        in_half_width = np.maximum(in_stopband, in_sampling_rate/2) - in_cutoff

        out_sampling_rate = out_size = in_size//2
        out_cutoff = out_sampling_rate/2
        out_stopband = out_cutoff * self.stopband_rel
        out_half_width = np.maximum(out_stopband, out_sampling_rate/2) - out_cutoff

        if self._in_size is not None and self._in_size == in_size:
            return
        else:
            self._initialize(
                in_size=in_size,
                out_size=out_size,
                in_sampling_rate=in_sampling_rate,
                out_sampling_rate=out_sampling_rate,
                in_cutoff=in_cutoff,
                out_cutoff=out_cutoff,
                in_half_width=in_half_width,
                out_half_width=out_half_width,
                padding=padding,
            )

    def _initialize(
        self,
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).
        padding=None,
    ):
        if padding is None:
            padding = self.filter_size // 2
        self._in_size = in_size
        self._out_size = out_size
        self.in_size = np.broadcast_to(np.asarray(in_size), [1])
        self.out_size = np.broadcast_to(np.asarray(out_size), [1])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = in_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = self.filter_size * self.down_factor if self.down_factor > 1 else 1
        if self.down_taps % 2 == 0:
            self.down_taps += 1
        self.down_filter = design_lowpass_filter_dct(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=False,
        ).to(self.device)

        # Compute padding.
        self.padding = padding

    def forward(self, x, **kwargs):
        self.initialize(in_size=x.shape[-1])
        x = upfirdn1d_dct(x=x, f=self.down_filter, down=self.down_factor, padding=self.padding,
                          dct=self.dct, idct=self.idct,
                          ) # Downsample.
        return x

class Upsample1d(nn.Module):
    def __init__(
        self,

        # Input & output specifications.
        scale_factor        = 2,

        # Hyperparameters.
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        padding             = None,
    ):
        super().__init__()
        assert np.exp2(int(np.log2(scale_factor))) == scale_factor, 'scale_factor is a power of two.'
        self.scale_factor = scale_factor
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        num_layers = int(np.log2(scale_factor))
        self.up = nn.ModuleList([_Upsample1d(
            stopband_rel=stopband_rel,
            filter_size=filter_size,
            padding=padding,
        ) for i in range(num_layers)])

    def forward(self, x, **kwargs):
        for m in self.up:
            x = m(x)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'scale_factor={self.scale_factor:g},',
            f'stopband_rel={self.stopband_rel:g},',
            f'filter_size={self.filter_size:d},',
            ])

class Downsample1d(nn.Module):
    def __init__(
        self,

        # Input & output specifications.
        scale_factor        = 1/2,

        # Hyperparameters.
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        padding             = None
    ):
        super().__init__()
        assert np.exp2(int(np.log2(1/scale_factor))) == 1/scale_factor, '1/scale_factor is a power of two.'
        self.scale_factor = scale_factor
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        num_layers = int(np.log2(1/scale_factor))
        self.down = nn.ModuleList([_Downsample1d(
            stopband_rel=stopband_rel,
            filter_size=filter_size,
            padding=padding,
        ) for i in range(num_layers)])

    def forward(self, x, **kwargs):
        for m in self.down:
            x = m(x)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'scale_factor={self.scale_factor:g},',
            f'stopband_rel={self.stopband_rel:g},',
            f'filter_size={self.filter_size:d},',
            ])

class _Upsample2d(nn.Module):
    def __init__(
        self,
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        use_radial  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        padding             = None,
    ):
        super().__init__()
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        self.use_radial = use_radial
        self.register_buffer('up_filter', None, persistent=False)
        self._in_size = None
        self.dct = LinearDCT(dim=2, dct_type='dct', norm=None)
        self.idct = LinearDCT(dim=2, dct_type='idct', norm=None)

    @property
    def device(self):
        up_filter = self.get_buffer('up_filter')
        if up_filter is None:
            return None
        else:
            return up_filter.device

    def initialize(self, in_size, padding=None):
        in_sampling_rate = in_size
        in_cutoff = in_sampling_rate/2
        in_stopband = in_cutoff * self.stopband_rel
        in_half_width = np.maximum(in_stopband, in_sampling_rate/2) - in_cutoff

        out_sampling_rate = out_size = 2*in_size
        out_cutoff = out_sampling_rate/2
        out_stopband = out_cutoff * self.stopband_rel
        out_half_width = np.maximum(out_stopband, out_sampling_rate/2) - out_cutoff

        if self._in_size is not None and self._in_size == in_size:
            return
        else:
            self._initialize(
                in_size=in_size,
                out_size=out_size,
                in_sampling_rate=in_sampling_rate,
                out_sampling_rate=out_sampling_rate,
                in_cutoff=in_cutoff,
                out_cutoff=out_cutoff,
                in_half_width=in_half_width,
                out_half_width=out_half_width,
                padding=padding,
            )

    def _initialize(
        self,
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).
        padding=None,
    ):
        if padding is None:
            padding = self.filter_size // 2
        self._in_size = in_size
        self._out_size = out_size
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        # self.up_taps = self.filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.up_taps = self.filter_size * self.up_factor if self.up_factor > 1 else 1
        if self.up_taps % 2 == 0:
            self.up_taps += 1
        self.up_radial = self.use_radial and not self.is_critically_sampled
        self.up_filter = design_lowpass_filter_dct(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate,
            radial=self.up_radial,
        ).to(self.device)

        # Compute padding.
        self.padding = padding

    def forward(self, x, **kwargs):
        self.initialize(in_size=x.shape[-1])
        x = upfirdn2d_dct(x=x, f=self.up_filter, up=self.up_factor, padding=self.padding,
                          gain=self.up_factor**2,
                          dct=self.dct, idct=self.idct,
                          ) # Upsample.
        return x

class _Downsample2d(nn.Module):
    def __init__(
        self,
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        use_radial  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        padding             = None,
    ):
        super().__init__()
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        self.use_radial = use_radial
        self.register_buffer('down_filter', None, persistent=False)
        self._in_size = None
        self.dct = LinearDCT(dim=2, dct_type='dct', norm=None)
        self.idct = LinearDCT(dim=2, dct_type='idct', norm=None)

    @property
    def device(self):
        down_filter = self.get_buffer('down_filter')
        if down_filter is None:
            return None
        else:
            return down_filter.device

    def initialize(self, in_size, padding=None):
        in_sampling_rate = in_size
        in_cutoff = in_sampling_rate/2
        in_stopband = in_cutoff * self.stopband_rel
        in_half_width = np.maximum(in_stopband, in_sampling_rate/2) - in_cutoff

        out_sampling_rate = out_size = in_size//2
        out_cutoff = out_sampling_rate/2
        out_stopband = out_cutoff * self.stopband_rel
        out_half_width = np.maximum(out_stopband, out_sampling_rate/2) - out_cutoff

        if self._in_size is not None and self._in_size == in_size:
            return
        else:
            self._initialize(
                in_size=in_size,
                out_size=out_size,
                in_sampling_rate=in_sampling_rate,
                out_sampling_rate=out_sampling_rate,
                in_cutoff=in_cutoff,
                out_cutoff=out_cutoff,
                in_half_width=in_half_width,
                out_half_width=out_half_width,
                padding=padding,
            )

    def _initialize(
        self,
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).
        padding=None,
    ):
        if padding is None:
            padding = self.filter_size // 2
        self._in_size = in_size
        self._out_size = out_size
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = in_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        # self.down_taps = self.filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_taps = self.filter_size * self.down_factor if self.down_factor > 1 else 1
        if self.down_taps % 2 == 0:
            self.down_taps += 1
        self.down_radial = self.use_radial and not self.is_critically_sampled
        self.down_filter = design_lowpass_filter_dct(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate,
            radial=self.down_radial,
        ).to(self.device)

        # Compute padding.
        self.padding = padding
        # print('padding: ', self.padding)

    def forward(self, x, **kwargs):
        self.initialize(in_size=x.shape[-1])
        x = upfirdn2d_dct(x=x, f=self.down_filter, down=self.down_factor, padding=self.padding,
                          dct=self.dct, idct=self.idct,
                          ) # Downsample.
        return x

class Upsample2d(nn.Module):
    def __init__(
        self,

        # Input & output specifications.
        scale_factor        = 2,

        # Hyperparameters.
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        use_radial  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        padding             = None,
    ):
        super().__init__()
        assert np.exp2(int(np.log2(scale_factor))) == scale_factor, 'scale_factor is a power of two.'
        self.scale_factor = scale_factor
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        self.use_radial = use_radial
        num_layers = int(np.log2(scale_factor))
        self.up = nn.ModuleList([_Upsample2d(
            stopband_rel=stopband_rel,
            filter_size=filter_size,
            use_radial=use_radial,
            padding=padding,
        ) for i in range(num_layers)])

    def forward(self, x, **kwargs):
        for m in self.up:
            x = m(x)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'scale_factor={self.scale_factor:g},',
            f'stopband_rel={self.stopband_rel:g},',
            f'filter_size={self.filter_size:d},',
            f'use_radial={self.use_radial},',
            ])

class Downsample2d(nn.Module):
    def __init__(
        self,

        # Input & output specifications.
        scale_factor        = 1/2,

        # Hyperparameters.
        stopband_rel        = 2**0.3,
        filter_size         = 7,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        use_radial  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        padding             = None
    ):
        super().__init__()
        assert np.exp2(int(np.log2(1/scale_factor))) == 1/scale_factor, '1/scale_factor is a power of two.'
        self.scale_factor = scale_factor
        self.stopband_rel = stopband_rel
        self.filter_size = filter_size
        self.use_radial = use_radial
        num_layers = int(np.log2(1/scale_factor))
        self.down = nn.ModuleList([_Downsample2d(
            stopband_rel=stopband_rel,
            filter_size=filter_size,
            use_radial=use_radial,
            padding=padding,
        ) for i in range(num_layers)])

    def forward(self, x, **kwargs):
        for m in self.down:
            x = m(x)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'scale_factor={self.scale_factor:g},',
            f'stopband_rel={self.stopband_rel:g},',
            f'filter_size={self.filter_size:d},',
            f'use_radial={self.use_radial},',
            ])

