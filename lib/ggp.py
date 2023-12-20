import functools
import scipy
import numpy as np
import torch
import torch.nn as nn

from lib.dct import LinearDCT


# def exponential_function(img_height, dim, exponent=2.0, length_scale=1.0, sigma=1.0, EPS=1e-5, device=None, **kwargs):
#     assert exponent > 0. and exponent <= 2.
#     assert length_scale > 0.
#     assert sigma > 0.
#
#     def get_lambda(v):
#         return (v / length_scale).abs() ** exponent

#     #freqs = np.pi * torch.linspace(0, img_height-1, img_height, device=device) / img_height
#     freqs = torch.linspace(0, img_height-1, img_height, device=device) / img_height
#     labda = get_lambda(freqs)
#     if dim == 1:
#         labda = labda[None, None, :]
#     elif dim == 2:
#         labda = labda[None, None, :, None] + labda[None, None, None, :]
#     elif dim == 3:
#         labda = labda[None, None, :, None, None] + labda[None, None, None, :, None] + labda[None, None, None, :, None]
#     else:
#         raise NotImplementedError

#     kernel = sigma**2 * torch.exp(-labda) + EPS
#     return kernel

# def exponential_kernel(img_height, dim, dct, exponent=2.0, length_scale=1.0, sigma=1.0, EPS=1e-5, device=None, **kwargs):
#     cov_func = exponential_function(
#         img_height=img_height,
#         dim=dim,
#         exponent=exponent,
#         length_scale=length_scale,
#         sigma=sigma,
#         EPS=EPS,
#         device=device)
#     cov_func_freq = dct(cov_func)
#     cov_func_freq[cov_func_freq <= 0] = 0
#     exp_kernel_freq = cov_func_freq ** 0.5

#     return exp_kernel_freq


def c_p(p):
    return p / 2**((p+1)/p) / scipy.special.gamma(1/p)

def u_p(x, p, EPS=1e-5):
    term1 = ( torch.sin(np.pi * x * p / 2.) / (torch.cos(np.pi * x / 2.) + EPS) )**(p/(1-p+EPS))
    term2 = torch.cos(np.pi * x * (p-1) / 2.) / (torch.cos(np.pi * x / 2.) + EPS)
    return term1 * term2

def g_p(t, x, p, EPS=1e-5):
    u = u_p(x, p)
    value = u * torch.exp( - t.abs()**(p/(p-1+EPS)) * u )
    value[torch.isnan(value)] = 0
    return value

def int_g_p(t, x, p, img_height=512, EPS=1e-5):
    return (g_p(t[...,None], x[None], p=p) / img_height).sum(dim=-1)

def _phi_p(t, p, t_th=0.01, img_height=512, EPS=1e-5):
    t = t.clone().detach()
    t[torch.logical_and(0 <= t, t < t_th)] = t_th
    t[torch.logical_and(0 > t, t > -t_th)] = -t_th
    x = torch.linspace(EPS, img_height-1, img_height, device=t.device) / img_height
    x[0] = 1e-4
    term1 = 2 * np.pi * p * t.abs()**(1/(p-1+EPS)) / (2 * abs(p-1) + EPS)
    term2 = int_g_p(t, x, p=p, img_height=img_height)
    return term1 * term2 #* c_p(p)

def phi_p(t, p, t_th=0.01, img_height=512, EPS=1e-5):
    if p == 1:
        return 2. / (1. + t**2)
    if 0.9 <= p and p <= 1.25 and t_th < 0.05:
        t_th = 0.05
    out = _phi_p(t=t, p=p, t_th=t_th, img_height=img_height, EPS=EPS)
    denom = _phi_p(t=torch.zeros_like(t), p=p, t_th=t_th, img_height=img_height, EPS=EPS)
    return out #/ denom

def exponential_kernel(img_height, dim, exponent=2.0, length_scale=1.0, sigma=1.0, device=None, **kwargs):
    freqs = np.pi * torch.linspace(0, img_height-1, img_height, device=device)
    cov_func_freq = phi_p(freqs * length_scale, p=exponent) * length_scale * (img_height-1)
    cov_func_freq /= (1. + length_scale * np.pi / 5.) # temporary

    if dim == 1:
        cov_func_freq = cov_func_freq[None, None, :]
    elif dim == 2:
        cov_func_freq = cov_func_freq[None, None, :, None] * cov_func_freq[None, None, None, :]
    elif dim == 3:
        cov_func_freq = cov_func_freq[None, None, :, None, None] * cov_func_freq[None, None, None, :, None] * cov_func_freq[None, None, None, :, None]
    else:
        raise NotImplementedError
    exp_kernel_freq = sigma * cov_func_freq ** 0.5

    return exp_kernel_freq


class RegularGridGaussianProcess(nn.Module):
    def __init__(self,
                 dim,
                 ch=None,
                 gp_type='exponential',
                 exponent=2.0,
                 length_scale=1.0,
                 sigma=1.0,
                 pos_range=(0., 1.),
                 #device=None,
                 modes=None,
                 **kwargs,
                 ):
        super().__init__()
        self.dim = dim
        self.pos_range = pos_range
        self.gp_type = self.kernel_type = gp_type
        self.modes = modes
        self.ch = ch

        if self.kernel_type == 'independent':
            self.dct = None
            self.idct = None
            self.sigma = sigma
        elif self.kernel_type == 'exponential':
            self.dct = LinearDCT(dim=dim, dct_type='dct', norm=None)
            self.idct = LinearDCT(dim=dim, dct_type='idct', norm=None)
            covfunc = exponential_kernel
            self.exponent = exponent
            self.length_scale = length_scale
            self.sigma = sigma
            self.covfunc = functools.partial(covfunc, dct=self.dct, exponent=exponent, length_scale=length_scale, sigma=sigma, **kwargs)
        else:
            raise NotImplementedError
        self.register_buffer('T', torch.Tensor([1.]))

        self.register_buffer('exp_kernel_freq', None, persistent=False)

    def extra_repr(self) -> str:
        desciption = f'(kernel_type): {self.kernel_type}'
        if self.kernel_type == 'independent':
            desciption += (f'\n'
                           f'(sigma): {self.sigma}'
                          )
        elif self.kernel_type == 'exponential':
            desciption += (f'\n'
                           f'(exponent): {self.exponent}\n'
                           f'(length_scale): {self.length_scale}\n'
                           f'(sigma): {self.sigma}\n'
                           f'(modes): {self.modes}'
                          )
        else:
            raise NotImplementedError
        return desciption

    def get_exp_kernel_freq(self, img_height, **kwargs):
        if self.exp_kernel_freq is None or self.exp_kernel_freq.shape[-1] != img_height:
            self.exp_kernel_freq = self.covfunc(img_height=img_height, dim=self.dim, device=self.T.device, **kwargs)
        if self.exp_kernel_freq.device != self.T.device:
            self.exp_kernel_freq = self.exp_kernel_freq.to(self.T.device)
        return self.exp_kernel_freq

    def sample_exp_kerenel(self, batch_size, img_height, ch, **kwargs):
        # init kernel
        exp_kernel_freq = self.get_exp_kernel_freq(img_height, **kwargs)

        # sample white noise
        if self.dim == 1:
            white_noise = torch.randn(batch_size, ch, img_height, device=exp_kernel_freq.device)
        elif self.dim == 2:
            white_noise = torch.randn(batch_size, ch, img_height, img_height, device=exp_kernel_freq.device)
        elif self.dim == 3:
            white_noise = torch.randn(batch_size, ch, img_height, img_height, img_height, device=exp_kernel_freq.device)

        # apply kernel
        out_noise_freq = exp_kernel_freq * self.dct(white_noise)

        # expand
        if self.modes is not None and self.modes < img_height:
            if self.dim == 1:
                out_noise_freq[:,:,self.modes:] = 0
            elif self.dim == 2:
                out_noise_freq[:,:,self.modes:] = 0
                out_noise_freq[:,:,:,self.modes:] = 0
            elif self.dim == 3:
                out_noise_freq[:,:,self.modes:] = 0
                out_noise_freq[:,:,:,self.modes:] = 0
                out_noise_freq[:,:,:,:,self.modes:] = 0

        # idct
        out_noise = self.idct(out_noise_freq)

        # return
        return out_noise

    def sample_indep_kernel(self, batch_size, img_height, ch, **kwargs):
        device = self.T.device

        # sample
        if self.dim == 1:
            out_noise = self.sigma * torch.randn(batch_size, ch, img_height, device=device)
        elif self.dim == 2:
            out_noise = self.sigma * torch.randn(batch_size, ch, img_height, img_height, device=device)
        elif self.dim == 3:
            out_noise = self.sigma * torch.randn(batch_size, ch, img_height, img_height, img_height, device=device)
        return out_noise

    def sample(self, batch_size, img_height, ch=None, **kwargs):
        if ch is not None:
            ch = ch
        else:
            ch = self.ch

        if self.kernel_type == 'independent':
            return self.sample_indep_kernel(batch_size=batch_size, img_height=img_height, ch=ch)
        elif self.kernel_type == 'exponential':
            return self.sample_exp_kerenel(batch_size=batch_size, img_height=img_height, ch=ch)
        else:
            raise NotImplementedError
