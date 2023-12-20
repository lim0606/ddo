from argparse import Namespace
import functools
import numpy as np
import torch
import torch.nn as nn
import tqdm

from thirdparty import torch_dct
from lib.dct import LinearDCT
from lib.ggp import RegularGridGaussianProcess


def low_discrepancy_rand(batch_size, device=None):
    u0 = torch.rand(1).item()
    t2 = ( u0 + torch.linspace(0, batch_size-1, batch_size, device=device) / batch_size ) % 1
    return t2

def low_discrepancy_randint(batch_size, k, device=None):
    u0 = torch.rand(1).item()
    t2 = torch.floor( ( u0 + torch.linspace(0, batch_size-1, batch_size, device=device) / batch_size ) * k % k ).long()
    return t2


# def get_noise_scaling_vp_linear(t, beta_min=0.1, beta_max=20., **kwargs):
#     alpha = torch.exp(-0.25 * t**2 * (beta_max-beta_min) - 0.5 * t * beta_min)
#     var = 1. - torch.exp(-0.5 * t**2 * (beta_max - beta_min) - t * beta_min)
#     sigma = var**0.5
# 
#     f =  - 0.5 * ( beta_min + (beta_max - beta_min) * t )
#     g2 = beta_min + (beta_max - beta_min) * t
#     g = g2**0.5
#     dvar_dt = alpha**2 * g2
#     return alpha, sigma, f, g, dvar_dt
# 
# def get_noise_scaling_cosine(t, logsnr_min=-10., logsnr_max=10., **kwargs):
#     limit_max = np.arctan(np.exp(-0.5 * logsnr_max))
#     limit_min = np.arctan(np.exp(-0.5 * logsnr_min)) - limit_max
#     logsnr = -2 * torch.log(torch.tan(limit_min * t + limit_max))
# 
#     # Transform logsnr to a, sigma.
#     alpha2 = torch.sigmoid(logsnr)
#     var = torch.sigmoid(-logsnr)
#     alpha = alpha2**0.5 #torch.sqrt(torch.sigmoid(logsnr))
#     sigma = var**0.5 #torch.sqrt(torch.sigmoid(-logsnr))
# 
#     coeff = limit_min / torch.tan(limit_min * t + limit_max) / torch.cos(limit_min * t + limit_max) ** 2
#     f = - (1. - alpha2) * coeff
#     dvar_dt =  2. * var * (1. - var) * coeff
#     g2 = dvar_dt - 2. * f * var
#     #g2 = -2. * f
#     g = g2**0.5
#     return alpha, sigma, f, g, dvar_dt

# def get_noise_scaling_ve_linear(t, sigma_min=1e-4, sigma_max=20., **kwargs):
#     return torch.ones_like(t), sigma_min + t * (sigma_max - sigma_min), None, None, None
# 
# def get_noise_scaling_ve_sine(t, sigma_min=1e-4, sigma_max=20., **kwargs):
#     return torch.ones_like(t), sigma_min + (sigma_max - sigma_min) * torch.sin(t * np.pi / 2)**2, None, None, None

# def get_noise_scaling(t, ns_method='vp_cosine', **kwargs):
#     if ns_method == 'vp_linear':
#         return get_noise_scaling_vp_linear(t, **kwargs)
#     elif ns_method == 'vp_cosine':
#         return get_noise_scaling_cosine(t, **kwargs)
# #     elif ns_method == 've_linear':
# #         return get_noise_scaling_ve_linear(t, **kwargs)
# #     elif ns_method == 've_sine':
# #         return get_noise_scaling_ve_sine(t, **kwargs)
#     else:
#         raise NotImplementedError


def sech(x):
    if isinstance(x, torch.Tensor):
        return 1. / torch.cosh(x)
    else:
        return 1. / np.cosh(x)

def shifted_sigmoid(lmbd, k:float):
    if isinstance(lmbd, torch.Tensor):
        return torch.sigmoid(-lmbd + k)
    else:
        return 1. / (1. + np.exp(lmbd - k))


def lmbd_cosine(t):
    if isinstance(t, torch.Tensor):
        return -2. * torch.log( torch.tan( 0.5 * np.pi * t ) )
    else:
        return -2. * np.log( np.tan( 0.5 * np.pi * t ) )

def inv_lmbd_cosine(lmbd):
    if isinstance(lmbd, torch.Tensor):
        return 2. * torch.arctan(torch.exp( - 0.5 * lmbd )) / np.pi
    else:
        return 2. * np.arctan(np.exp( - 0.5 * lmbd )) / np.pi

def pdf_cosine(lmbd):
    return sech(0.5 * lmbd) / ( 2. * np.pi )

def dlmbd_dt_cosine(t):
    return - np.pi / ( torch.tan( 0.5 * np.pi * t ) * torch.cos( 0.5 * np.pi * t )**2 )

def get_t0_t1(lmbd0:float, lmbd1:float, inv_lmbd_fn):
    t0 = inv_lmbd_fn(lmbd0)
    t1 = inv_lmbd_fn(lmbd1)
    return t0, t1

def truncated_lmbd(t:torch.Tensor, t0:float, t1:float, lmbd_fn):
    return lmbd_fn(t0 + (t1 - t0) * t)

def truncated_inv_lmbd(lmbd:torch.Tensor, t0:float, t1:float, inv_lmbd_fn):
    return (inv_lmbd_fn(lmbd) - t0) / (t1 - t0)

def truncated_pdf(lmbd:torch.Tensor, lmbd0:float, lmbd1:float, t0:float, t1:float, pdf_fn):
    mask = (lmbd1 <= lmbd).float() * (lmbd <= lmbd0).float()
    return mask * pdf_fn(lmbd) / (t1 - t0)

def truncated_dlmbd_dt(t:torch.Tensor, t0:float, t1:float, dlmbd_dt_fn):
    return dlmbd_dt_fn(t0 + (t1 - t0) * t) * (t1 - t0)

def normalize_lmbd(lmbd, lmbd0, lmbd1):
    assert lmbd0 > lmbd1
    return (lmbd - lmbd1) / (lmbd0 - lmbd1)

def get_vp_alpha_sigma_from_lmbd(lmbd:torch.Tensor):
    alpha = torch.sigmoid(lmbd) ** 0.5
    sigma = torch.sigmoid(-lmbd) ** 0.5
    return alpha, sigma

def get_vp_f_g_from_lmbd(lmbd:torch.Tensor, inv_lmbd_fn, dlmbd_dt_fn):
    dlmbd_dt = dlmbd_dt_fn(inv_lmbd_fn(lmbd))
    f = 0.5 * torch.sigmoid( -lmbd ) * dlmbd_dt
    g = (- torch.sigmoid( -lmbd ) * dlmbd_dt)**0.5
    return f, g

def get_vp_noise_scaling_from_lmbd(lmbd:torch.Tensor, inv_lmbd_fn, dlmbd_dt_fn, return_all=False):
    alpha, sigma = get_vp_alpha_sigma_from_lmbd(lmbd)
    f, g = get_vp_f_g_from_lmbd(lmbd, inv_lmbd_fn=inv_lmbd_fn, dlmbd_dt_fn=dlmbd_dt_fn)
    return alpha, sigma, f, g

#def get_noise_scaling(t, ns_method='vp_cosine', lmbd0=20, lmbd1=-20, return_all=False, **kwargs):
def get_noise_scaling(t, ns_method='vp_cosine', lmbd0=10, lmbd1=-10, return_all=False, **kwargs):
    if ns_method == 'vp_cosine':
        # init fns
        t0, t1 = get_t0_t1(lmbd0, lmbd1, inv_lmbd_cosine)
        truncated_lmbd_cosine = functools.partial(truncated_lmbd, t0=t0, t1=t1, lmbd_fn=lmbd_cosine)
        truncated_inv_lmbd_cosine = functools.partial(truncated_inv_lmbd, t0=t0, t1=t1, inv_lmbd_fn=inv_lmbd_cosine)
        truncated_pdf_cosine = functools.partial(truncated_pdf, lmbd0=lmbd0, lmbd1=lmbd1, t0=t0, t1=t1, pdf_fn=pdf_cosine)
        truncated_dlmbd_dt_cosine = functools.partial(truncated_dlmbd_dt, t0=t0, t1=t1, dlmbd_dt_fn=dlmbd_dt_cosine)

        # get lmbd
        lmbd = truncated_lmbd_cosine(t)
        lmbd_normalized = normalize_lmbd(lmbd, lmbd0=lmbd0, lmbd1=lmbd1)
        pdf = truncated_pdf_cosine(lmbd)
        alpha, sigma, f, g = get_vp_noise_scaling_from_lmbd(lmbd,
                                                            inv_lmbd_fn=truncated_inv_lmbd_cosine,
                                                            dlmbd_dt_fn=truncated_dlmbd_dt_cosine,
                                                            )
    else:
        raise NotImplementedError 
    if return_all:
        return lmbd, lmbd_normalized, pdf, alpha, sigma, f, g, None
    else:
        return alpha, sigma, f, g, None

def get_weight(lmbd, weight_method='shifted_sigmoid_2'):
    if weight_method == 'shifted_sigmoid_1':
        return shifted_sigmoid(lmbd, k=1)
    elif weight_method == 'shifted_sigmoid_2':
        return shifted_sigmoid(lmbd, k=2)
    elif weight_method == 'shifted_sigmoid_3':
        return shifted_sigmoid(lmbd, k=3)
    else:
        raise NotImplementedError


def get_sigma_t_linear(t, sigma_blur_min=0.5, sigma_blur_max=20., **kwargs):
    return sigma_blur_min + t * (sigma_blur_max - sigma_blur_min)

def get_dsigma_dt_linear(t, sigma_blur_min=0.5, sigma_blur_max=20., **kwargs):
    return torch.ones_like(t) * (sigma_blur_max - sigma_blur_min)


def get_sigma_t_sine(t, sigma_blur_min=0., sigma_blur_max=20.0, **kwargs):
    return sigma_blur_min + (sigma_blur_max - sigma_blur_min) * torch.sin(t * np.pi / 2)**2

def get_dsigma_dt_sine(t, sigma_blur_min=0., sigma_blur_max=20.0, **kwargs):
    return (sigma_blur_max - sigma_blur_min) * np.pi * torch.sin(t * np.pi / 2) * torch.cos(t * np.pi / 2)


def get_sigma_t_log(t, sigma_blur_min=0.5, sigma_blur_max=20., **kwargs):
    log_sigma_blur_max = np.log(sigma_blur_max)
    log_sigma_blur_min = np.log(sigma_blur_min)
    return sigma_blur_min * torch.exp(t * (log_sigma_blur_max - log_sigma_blur_min))

def get_dsigma_dt_log(t, sigma_blur_min=0.5, sigma_blur_max=20., **kwargs):
    log_sigma_blur_max = np.log(sigma_blur_max)
    log_sigma_blur_min = np.log(sigma_blur_min)
    sigma_t = torch.exp(log_sigma_blur_min + t * (log_sigma_blur_max - log_sigma_blur_min))
    return sigma_t * (log_sigma_blur_max - log_sigma_blur_min)


# def get_sigma_t_const(t, sigma_blur=1.0, **kwargs):
#     return sigma_blur * torch.ones_like(t)

# def get_dsigma_dt_const(t, sigma_blur=1.0, **kwargs):
#     return torch.zeros_like(t)


def get_sigma_t(t, disp_method='linear', **kwargs):
    # if disp_method == 'const':
    #     return get_sigma_t_const(t, **kwargs)
    if disp_method == 'linear':
        return get_sigma_t_linear(t, **kwargs)
    elif disp_method == 'log':
        return get_sigma_t_log(t, **kwargs)
    elif disp_method == 'sine':
        return get_sigma_t_sine(t, **kwargs)
    else:
        raise NotImplementedError

def get_dsigma_dt(t, disp_method='linear', **kwargs):
    # if disp_method == 'const':
    #     return get_dsigma_dt_const(t, **kwargs)
    if disp_method == 'linear':
        return get_dsigma_dt_linear(t, **kwargs)
    elif disp_method == 'log':
        return get_dsigma_dt_log(t, **kwargs)
    elif disp_method == 'sine':
        return get_dsigma_dt_sine(t, **kwargs)
    else:
        raise NotImplementedError


def get_frequency_scaling(t, img_dim, disp_method='linear', dim:int=1, **kwargs):
    # assert min_scale <= 1. and min_scale >= 0.
    # init
    t = t.view(-1)

    # compute dissipation time
    sigma_t = get_sigma_t(t, disp_method=disp_method, **kwargs)
    dsigma_dt = get_dsigma_dt(t, disp_method=disp_method, **kwargs)
    dissipation_time = sigma_t**2 / 2.
    dissipation_time_derivative = sigma_t * dsigma_dt

    # compute frequencies
    # freqs = np.pi * torch.linspace(0, img_dim-1, img_dim, device=t.device) / img_dim
    freqs = torch.linspace(0, img_dim-1, img_dim, device=t.device)
    if dim == 1:
        labda = freqs[None, None, :]**2
        dissipation_time = dissipation_time[..., None, None]
        dissipation_time_derivative = dissipation_time_derivative[..., None, None]
    elif dim == 2:
        labda = freqs[None, None, :, None]**2 + freqs[None, None, None, :]**2
        dissipation_time = dissipation_time[..., None, None, None]
        dissipation_time_derivative = dissipation_time_derivative[..., None, None, None]
    else:
        raise NotImplementedError

    # compute scaling for frequencies
    # scaling = torch.exp(-labda * dissipation_time) * (1 - min_scale) + min_scale
    # scaling_tilde = -labda * dissipation_time_derivative / ( 1. + min_scale / torch.exp(-labda * dissipation_time) / (1. - min_scale + 1e-8))
    scaling = torch.exp(-labda * dissipation_time)
    scaling_tilde = -labda * dissipation_time_derivative * scaling
    return scaling, scaling_tilde


def expand_dim(t, dim=None):
    t = t.view(-1)
    if dim is None or dim == 0:
        return t
    elif dim == 1:
        return t[:, None, None]
    elif dim == 2:
        return t[:, None, None, None]
    elif dim == 3:
        return t[:, None, None, None, None]
    else:
        raise NotImplementedError


def get_lmbd_pdf(t, dim:int=1, ns_method='vp_cosine', **kwargs):
    # init
    t = t.view(-1)
    lmbd, lmbd_normalized, pdf, _, _, _, _, _ = get_noise_scaling(t, ns_method=ns_method, return_all=True, **kwargs)
    return expand_dim(lmbd, dim), expand_dim(lmbd_normalized), expand_dim(pdf, dim)


def get_alpha_sigma(t, dim:int=1, ns_method='vp_cosine', **kwargs):
    # init
    t = t.view(-1)
    alpha, sigma, _, _, _ = get_noise_scaling(t, ns_method=ns_method, **kwargs)
    return expand_dim(alpha, dim), expand_dim(sigma, dim)


def get_alpha_tilde_sigma_tilde(t, img_dim, dim:int=1, disp_method='const', ns_method='vp_cosine', **kwargs):
    # init
    t = t.view(-1)
    freq_scaling, _ = get_frequency_scaling(t, img_dim, disp_method=disp_method, dim=dim, **kwargs)
    a, sigma, _, _, _ = get_noise_scaling(t, ns_method=ns_method, **kwargs)
    alpha = expand_dim(a, dim) * freq_scaling
    sigma = expand_dim(sigma, dim)
    return alpha, sigma


def get_f_g(t, img_dim, dim:int=1, disp_method='const', ns_method='vp_cosine', **kwargs):
    # init
    t = t.view(-1)
    _, sigma, f, g, _ = get_noise_scaling(t, ns_method=ns_method, **kwargs)
    return expand_dim(f, dim), expand_dim(g, dim)


def get_f_tilde_g_tilde(t, img_dim, dim:int=1, disp_method='const', ns_method='vp_cosine', **kwargs):
    # init
    t = t.view(-1)
    _, freq_scaling_tilde = get_frequency_scaling(t, img_dim, disp_method=disp_method, dim=dim, **kwargs)
    #print(freq_scaling_tilde)
    _, sigma, f, g, dvar_dt = get_noise_scaling(t, ns_method=ns_method, **kwargs)
    f_tilde = expand_dim(f, dim) + freq_scaling_tilde
    #g2_tilde = expand_dim(dvar_dt, dim) - 2. * f_tilde * expand_dim(sigma**2, dim)
    g2_tilde = expand_dim(g**2, dim) - 2. * freq_scaling_tilde * expand_dim(sigma**2, dim)
    return f_tilde, g2_tilde ** 0.5
    # return f_tilde, expand_dim(g, dim)


class LinearDiffusion(nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self,
                 dim,
                 ch,
                 ns_method:str='vp_cosine',
                 ns_config:Namespace=Namespace(),
                 gp_type:str='independent',
                 gp_config:Namespace=Namespace(),
                 **kwargs,
                ):
        super().__init__()
        self.ns_method = ns_method
        self.gp_type = gp_type
        self.register_buffer('T', torch.Tensor([1.]))
        self.dim = dim
        self.ch = ch

        ns_config.ns_method = ns_method
        self.ns_config = ns_config
        gp_config.gp_type = gp_type
        gp_config.dim = dim
        gp_config.ch = ch
        self.gp_config = gp_config
        if gp_type in ['independent', 'exponential']:
            self.gp = RegularGridGaussianProcess(**vars(self.gp_config))
        else:
            raise NotImplementedError

        self.all_kwargs = vars(self.ns_config).copy()

        self.dct = nn.Identity()
        self.idct = nn.Identity()

    def extra_repr(self) -> str:
        return (f'(dim): {self.dim}\n'
                f'(ns): {self.ns_config}\n'
                f'(gp): {self.gp_type}'
                )

    def get_f_g(self, t, img_dim):
        return get_f_g(t, img_dim=img_dim, dim=self.dim, **self.all_kwargs)

    def forward_step(self, z_t, t, dt, v=None, return_all=False, **kwargs):
        # init
        t = t.view(-1)
        img_dim = z_t.shape[-1]
        assert self.ch == z_t.shape[1]

        # Get f and g
        f, g = self.get_f_g(t, img_dim)

        # noise
        eps = self.gp.sample(z_t.shape[0], z_t.shape[-1]).to(z_t.device)

        # update
        mu = z_t + dt * f * z_t
        sigma_eps = dt ** 0.5 * g * eps
        z_tp1 = mu + sigma_eps

        if return_all:
            return z_tp1, eps, mu, sigma_eps
        else:
            return z_tp1

    def get_lmbd_pdf(self, t, dim=None):
        return get_lmbd_pdf(t,
                            dim=self.dim if dim is None else dim,
                            **self.all_kwargs,
                            )

    def get_alpha_sigma(self, t, img_dim):
        return get_alpha_sigma(t,
                               img_dim=img_dim,
                               dim=self.dim,
                               #ns_method=self.ns_method,
                               #disp_method=self.disp_method,
                               **self.all_kwargs,
                              )

    def diffuse_step(self, z_t, t, num_steps, v=None, return_all=False, **kwargs):
        # init
        t = t.view(-1)
        img_dim = z_t.shape[-1]

        # Get alpha and sigma
        alpha_s, sigma_s = self.get_alpha_sigma(t, img_dim=img_dim)
        alpha_t, sigma_t = self.get_alpha_sigma(t + 1 / num_steps, img_dim=img_dim)

        # Compute helpful coefficients.
        alpha_ts = alpha_t / alpha_s
        sigma2_ts = (sigma_t**2 - alpha_ts**2 * sigma_s**2)

        # Compute terms.
        mu = alpha_ts * z_t

        # Sample from the denoising distribution (zero mean).
        eps = self.gp.sample(z_t.shape[0], z_t.shape[-1]).to(z_t.device)
        sigma_eps = torch.sqrt(sigma2_ts) * eps

        # Sample from the denoising distribution.
        z_tm1 = mu + sigma_eps

        # Return
        if return_all:
            return z_tm1, eps, mu, sigma_eps
        else:
            return z_tm1

    def diffuse(self, x, t, **kwargs):
        # init
        t = t.view(-1)
        if self.dim == 1:
            assert x.dim() == 3 # B, C, H
        elif self.dim == 2:
            assert x.dim() == 4 # B, C, H, W
        else:
            raise NotImplementedError

        # get alpha and sigma
        alpha, sigma = self.get_alpha_sigma(t, img_dim=x.shape[-1])

        # noise
        eps = self.gp.sample(x.shape[0], x.shape[-1]).to(x.device)

        # Since we chose sigma to be a scalar, eps does not need to be
        # passed through a DCT/IDCT in this case.
        mu_diffuse =  alpha * x
        z_t = mu_diffuse + sigma * eps

        # return
        return z_t, eps, mu_diffuse, sigma

    def sample(self, t, x0, **kwargs):
        """
        sample xt | x0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        xt, epsilon, mu, sigma = self.diffuse(x=x0, t=t, **kwargs)
        return xt, epsilon, mu, sigma


class BlurringDiffusion(nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self,
                 dim,
                 ch,
                 ns_method:str='vp_cosine',
                 ns_config:Namespace=Namespace(),
                 disp_method:str='linear',
                 disp_config:Namespace=Namespace(**{
                     #'min_scale': 0.,
                     'sigma_blur_min': 0.,
                     'sigma_blur_max': 0.,
                 }),
                 gp_type:str='independent',
                 gp_config:Namespace=Namespace(),
                 **kwargs,
                ):
        super().__init__()
        self.ns_method = ns_method
        self.disp_method = disp_method
        self.gp_type = gp_type
        self.register_buffer('T', torch.Tensor([1.]))
        self.dim = dim
        self.ch = ch

        ns_config.ns_method = ns_method
        self.ns_config = ns_config
        disp_config.disp_method = disp_method
        self.disp_config = disp_config
        gp_config.gp_type = gp_type
        gp_config.dim = dim
        gp_config.ch = ch
        self.gp_config = gp_config
        if gp_type in ['independent', 'exponential']:
            # self.gp = GaussianRF(**self.gp_kwargs)
            self.gp = RegularGridGaussianProcess(**vars(self.gp_config))
        else:
            raise NotImplementedError

        self.all_kwargs = vars(self.ns_config).copy()
        self.all_kwargs.update(vars(self.disp_config).copy())

        # if disp_config.min_scale == 0. \
        #     and disp_config.sigma_blur_min == 0. \
        #     and disp_config.sigma_blur_max == 0.:
        if disp_config.sigma_blur_min == 0. and disp_config.sigma_blur_max == 0.:
            self.dct = nn.Identity()
            self.idct = nn.Identity()
        else:
            self.dct = LinearDCT(dim=self.dim, dct_type='dct')
            self.idct = LinearDCT(dim=self.dim, dct_type='idct')

    def extra_repr(self) -> str:
        return (f'(dim): {self.dim}\n'
                f'(ns): {self.ns_config}\n'
                f'(disp): {self.disp_config}\n'
                f'(gp): {self.gp_type}'
                )

    def get_f_g(self, t, img_dim):
        return get_f_tilde_g_tilde(t,
                                   img_dim=img_dim,
                                   dim=self.dim,
                                   #ns_method=self.ns_method,
                                   #disp_method=self.disp_method,
                                   **self.all_kwargs,
                                   )

    def forward_step(self, z_t, t, dt, v=None, return_all=False, **kwargs):
        # init
        t = t.view(-1)
        img_dim = z_t.shape[-1]
        assert self.ch == z_t.shape[1]

        # Get f and g
        f, g = self.get_f_g(t, img_dim)

        # get alpha0
        alpha0, _ = self.get_alpha_sigma(torch.zeros_like(t[:1]), img_dim=z_t.shape[-1])
        t_mask = expand_dim(t < 1e-4, self.dim).float().expand_as(f)
        alpha0 = torch.ones_like(f) * (1.-t_mask) + alpha0 * t_mask

        # input
        u_t = alpha0 * self.dct(z_t)

        # noise
        eps = self.gp.sample(z_t.shape[0], z_t.shape[-1]).to(z_t.device)
        if self.gp.gp_type == 'independent':
            u_eps = eps
        else:
            u_eps = self.dct(eps)

        # update
        u_mu = u_t + dt * f * u_t
        u_sigma_eps = dt ** 0.5 * g * u_eps
        u_tp1 = u_mu + u_sigma_eps

        # one step update of Euler Maruyama method with a step size dt
        z_tp1 = self.idct(u_tp1)

        if return_all:
            return z_tp1, eps, self.idct(u_mu), self.idct(u_sigma_eps)
        else:
            return z_tp1

    def get_alpha_sigma(self, t, img_dim):
        return get_alpha_tilde_sigma_tilde(t,
                                           img_dim=img_dim,
                                           dim=self.dim,
                                           #ns_method=self.ns_method,
                                           #disp_method=self.disp_method,
                                           **self.all_kwargs,
                                           )

    def diffuse_step(self, z_t, t, num_steps, v=None, return_all=False, **kwargs):
        # init
        t = t.view(-1)
        img_dim = z_t.shape[-1]

        # Get alpha and sigma
        alpha_s, sigma_s = self.get_alpha_sigma(t, img_dim=img_dim)
        alpha_t, sigma_t = self.get_alpha_sigma(t + 1 / num_steps, img_dim=img_dim)

        # get alpha0
        alpha0, _ = self.get_alpha_sigma(torch.zeros_like(t[:1]), img_dim=z_t.shape[-1])
        t_mask = expand_dim(t < 1e-4, self.dim).float().expand_as(alpha_t)
        alpha0 = torch.ones_like(alpha_t) * (1.-t_mask) + alpha0 * t_mask

        # Compute helpful coefficients.
        alpha_ts = alpha_t / alpha_s
        sigma2_ts = (sigma_t**2 - alpha_ts**2 * sigma_s**2)

        # Compute terms.
        u_t = alpha0 * self.dct(z_t)
        mu = self.idct(alpha_ts * u_t)

        # Sample from the denoising distribution (zero mean).
        eps = self.gp.sample(z_t.shape[0], z_t.shape[-1]).to(z_t.device)
        if self.gp.gp_type == 'independent':
            u_eps = eps
        else:
            u_eps = self.dct(eps)
        sigma_eps = self.idct(torch.sqrt(sigma2_ts) * u_eps)

        # Sample from the denoising distribution.
        z_tm1 = mu + sigma_eps

        # Return
        if return_all:
            return z_tm1, eps, mu, sigma_eps
        else:
            return z_tm1

    def diffuse(self, x, t, **kwargs):
        # init
        t = t.view(-1)
        if self.dim == 1:
            assert x.dim() == 3 # B, C, H
        elif self.dim == 2:
            assert x.dim() == 4 # B, C, H, W
        else:
            raise NotImplementedError

        # dct
        x_freq = self.dct(x)

        # get alpha and sigma
        alpha, sigma = self.get_alpha_sigma(t, img_dim=x.shape[-1])

        # noise
        eps = self.gp.sample(x.shape[0], x.shape[-1]).to(x.device)

        # Since we chose sigma to be a scalar, eps does not need to be
        # passed through a DCT/IDCT in this case.
        mu_diffuse = self.idct(alpha * x_freq)
        z_t = mu_diffuse + sigma * eps

        # return
        return z_t, eps, mu_diffuse, sigma

    def sample(self, t, x0, **kwargs):
        """
        sample xt | x0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        xt, epsilon, mu, sigma = self.diffuse(x=x0, t=t, **kwargs)
        return xt, epsilon, mu, sigma


class DenoisingDiffusion(nn.Module):
    def __init__(self, forward_diffusion, model, model_type="epsilon", timestep_sampler="uniform", use_clip=False, weight_method=None, use_lmbd=False):
        super().__init__()
        self.forward_diffusion = forward_diffusion
        assert model_type in ["epsilon"]
        self.model_type = model_type
        self._model = model
        self.use_lmbd = use_lmbd
        assert timestep_sampler in ["uniform", "low_discrepancy"]
        self.timestep_sampler = timestep_sampler
        self.use_clip = use_clip
        self.weight_method = weight_method

    # model
    def model(self, x, temp, v=None, level=None):
        if self.use_lmbd:
            _, lmbd_normalized, _ = self.get_lmbd_pdf(temp, dim=0)
            return self._model(x=x, temp=lmbd_normalized, v=v, level=level)
        else:
            return self._model(x=x, temp=temp, v=v, level=level)

    # epsilon prediction
    def epsilon(self, y, s, v=None, level=None, **kwargs):
        if self.model_type == 'epsilon':
            return self.model(x=y, temp=s.view(-1), v=v, level=level)
        else:
            raise NotImplementedError

    # score prediction
    def score(self, y, s, v=None, level=None, **kwargs):
        if self.model_type == 'epsilon':
            _, sigma = self.forward_diffusion.get_alpha_sigma(s, img_dim=y.shape[-1])
            return - self.model(x=y, temp=s.view(-1), v=v, level=level) / sigma
        else:
            raise ValueError

    # data prediction
    def pred(self, y, s, v=None, level=None, **kwargs):
        if self.model_type == 'epsilon':
            alpha, sigma = self.forward_diffusion.get_alpha_sigma(s, img_dim=y.shape[-1])
            epsilon = self.model(x=y, temp=s.view(-1), v=v, level=level)
            return (y - sigma * epsilon) / alpha
        else:
            raise ValueError

    # helper function
    def _clipped_epsilon(self, y, s, v=None, level=None, **kwargs):
        alpha, sigma = self.forward_diffusion.get_alpha_sigma(s, img_dim=y.shape[-1])
        pred = self.pred(y=y, s=s, v=v, level=level, **kwargs).clip(-1, 1)
        return (y - alpha * pred) / sigma

    # helper function
    def _clipped_score(self, y, s, v=None, level=None, **kwargs):
        alpha, sigma = self.forward_diffusion.get_alpha_sigma(s, img_dim=y.shape[-1])
        pred = self.pred(y=y, s=s, v=v, level=level, **kwargs).clip(-1, 1)
        return - (y - alpha * pred) / sigma**2

    # forward step
    def forward_step(self, z_t, t, dt, v=None, lmbd=0., return_all=False, use_clip=None, **kwargs):
        # init
        t = t.view(-1)
        img_dim = z_t.shape[-1]

        # Get f and g
        f, g = self.forward_diffusion.get_f_g(self.T - t, img_dim)

        # Get f_tilde and g_tilde
        if use_clip is None:
            use_clip = self.use_clip
        if use_clip:
            score = self._clipped_score(y=z_t, s=self.T - t, v=v)
        else:
            score = self.score(y=z_t, s=self.T - t, v=v)
        score_freq = self.dct(score)
        u_t = self.dct(z_t)
        f_tilde = (1. - 0.5 * lmbd) * self.idct(g**2 * score_freq) - self.idct(f * u_t)
        g_tilde = (1. - lmbd) ** 0.5 * g

        # mu
        mu = z_t + dt * f_tilde

        # noise
        eps = self.gp.sample(z_t.shape[0], z_t.shape[-1]).to(z_t.device)
        if self.gp.gp_type == 'independent':
            u_eps = eps
        else:
            u_eps = self.dct(eps)
        sigma_eps = dt ** 0.5 * self.idct(g_tilde * u_eps)

        # one step update of Euler Maruyama method with a step size dt
        z_tp1 = mu + sigma_eps

        if return_all:
            return z_tp1, eps, mu, sigma_eps
        else:
            return z_tp1

    @torch.enable_grad()
    def dsm(self, z0, v=None, **kwargs):
        """
        denoising score matching loss
        """
        if self.timestep_sampler == "uniform":
            s_ = torch.rand(z0.size(0), device=z0.device) * self.T
        elif self.timestep_sampler == "low_discrepancy":
            # batch_size = z0.shape[0]
            # u0 = torch.rand(1).item()
            # s_ = ( (u0 + torch.linspace(0, batch_size-1, batch_size, device=z0.device) / batch_size) % 1 ) * self.T
            s_ = low_discrepancy_rand(z0.shape[0], device=z0.device) * self.T
        else:
            raise ValueError

        zt, target, _, _ = self.forward_diffusion.sample(t=s_, x0=z0)
        pred = self.epsilon(y=zt, s=s_, v=v)
        mse = 0.5 * ((pred - target) ** 2)
        if self.weight_method is not None:
            lmbd, _, pdf = self.get_lmbd_pdf(s_)
            weight = get_weight(lmbd, weight_method=self.weight_method)
            mse = weight / pdf * mse
        return mse.view(z0.size(0), -1).sum(1, keepdim=False)

    @property
    def in_channels(self):
        return self._model.in_channels

    @property
    def dim(self):
        return self.forward_diffusion.dim

    @property
    def T(self):
        return self.forward_diffusion.T

    @property
    def device(self):
        return self.T.device

    def dct(self, *args, **kwargs):
        return self.forward_diffusion.dct(*args, **kwargs)

    def idct(self, *args, **kwargs):
        return self.forward_diffusion.idct(*args, **kwargs)

    @property
    def gp(self):
        return self.forward_diffusion.gp

    def diffuse(self, *args, **kwargs):
        return self.forward_diffusion.diffuse(*args, **kwargs)

    def get_lmbd_pdf(self, *args, **kwargs):
        return self.forward_diffusion.get_lmbd_pdf(*args, **kwargs)

    def get_alpha_sigma(self, *args, **kwargs):
        return self.forward_diffusion.get_alpha_sigma(*args, **kwargs)

    def denoise_step(self, z_t, t, num_steps, v=None, delta=1e-8, return_all=False, use_clip=None, **kwargs):
        # init
        t = t.view(-1)
        img_dim = z_t.shape[-1]
        # assert num_steps >= 100, 'num_steps < 100 is inaccurate'
        assert torch.all(t - 1 / num_steps >= 0).item()

        # Get alpha and sigma
        alpha_s, sigma_s = self.get_alpha_sigma(t - 1 / num_steps, img_dim=img_dim)
        alpha_t, sigma_t = self.get_alpha_sigma(t, img_dim=img_dim)
        sigma2_s = sigma_s**2
        sigma2_t = sigma_t**2

        # Compute helpful coefficients.
        alpha_ts = alpha_t / torch.clip(alpha_s, min=delta)
        alpha_st = alpha_s / torch.clip(alpha_t, min=delta)
        #sigma2_ts = (sigma_t**2 - alpha_ts**2 * sigma_s**2)
        sigma2_ts = (sigma2_t - alpha_ts**2 * sigma2_s)

        # Denoising variance.
        # sigma2_denoise = 1 / torch.clip(
        #     1 / torch.clip(sigma_s**2, min=delta) +
        #     1 / torch.clip(sigma_t**2 / alpha_ts**2 - sigma_s**2, min=delta),
        #     min=delta)
        sigma2_denoise = sigma2_ts * sigma2_s / torch.clip(sigma2_t, min=delta)

        # The coefficients for u_t and u_eps.
        # coeff_term1 = alpha_ts * sigma2_denoise / (sigma2_ts + delta)
        # coeff_term2 = alpha_st * sigma2_denoise / torch.clip(
        #     alpha_ts, min=delta) / torch.clip(sigma_s**2, min=delta)
        coeff_term1 = alpha_ts * sigma2_s / torch.clip(sigma2_t, min=delta)
        coeff_term2 = alpha_st * sigma2_ts / torch.clip(sigma2_t, min=delta)

        # Get neural net prediction.
        if use_clip is None:
            use_clip = self.use_clip
        if use_clip:
            hat_eps = self._clipped_epsilon(y=z_t, s=t, v=v)
        else:
            hat_eps = self.epsilon(y=z_t, s=t, v=v)

        # Compute terms.
        u_t = self.dct(z_t)
        term1 = self.idct(coeff_term1 * u_t)
        term2 = self.idct(coeff_term2 * (u_t - sigma_t * self.dct(hat_eps)))
        mu_denoise = term1 + term2

        # Sample from the denoising distribution (zero mean).
        eps = self.gp.sample(z_t.shape[0], z_t.shape[-1]).to(z_t.device)
        if self.gp.gp_type == 'independent':
            u_eps = eps
        else:
            u_eps = self.dct(eps)
        sigma_eps_denoise = self.idct(torch.sqrt(sigma2_denoise) * u_eps)

        # Sample from the denoising distribution.
        z_tm1 = mu_denoise + sigma_eps_denoise

        # Return
        if return_all:
            return z_tm1, eps, mu_denoise, sigma_eps_denoise
            # return z_tm1, eps, mu_denoise, (z_tm1, eps, alpha_s, alpha_t, sigma2_s, sigma2_t, sigma2_ts, sigma2_denoise, coeff_term1, coeff_term2, z_t, hat_eps)
        else:
            return z_tm1


@torch.no_grad()
def diffuse(model, num_steps, x_0, v=None, lmbd=0., s_min=0., sampler='denoise', return_mean=True, keep_all_samples=False, disable_tqdm=True, use_clip=None, **kwargs):
    # init
    model.eval()
    device = x_0.device
    xsz = list(x_0.shape)
    ONES = torch.ones(xsz[0],).cuda()
    dt = None

    # init sampler
    if sampler == 'denoise':
        ts = np.linspace(num_steps, 1, num_steps) / num_steps
        func_step = getattr(model, 'denoise_step')
    elif  sampler == 'diffuse':
        ts = np.linspace(0, num_steps-1, num_steps) / num_steps
        func_step = getattr(model, 'diffuse_step')
    elif sampler == 'em':
        T_ = model.T.item()
        dt = (T_-s_min) / (num_steps + 1)
        ts = np.linspace(0., T_-s_min-dt, num_steps)
        func_step = getattr(model, 'forward_step')
    else:
        raise NotImplementedError

    # init x_t
    x_t = x_0.detach().clone().cuda()
    if keep_all_samples:
        xs = [x_t.detach().clone().to('cpu')[None]]
    else:
        xs = []

    # sample
    n_ts = len(ts)
    with torch.no_grad() as _ , tqdm.tqdm(disable=disable_tqdm, position=0, leave=False, unit='steps', total=n_ts, desc='Sampling', ncols=80) as tm:
        for i in range(n_ts):
            # init t
            t = ts[i] * ONES

            if return_mean and i == n_ts-1:
                _, _, mu_t, _ = func_step(x_t, t, v=v, num_steps=num_steps, dt=dt, lmbd=lmbd, return_all=True, use_clip=use_clip)
                x_t = mu_t
            else:
                x_t, _, _, _  = func_step(x_t, t, v=v, num_steps=num_steps, dt=dt, lmbd=lmbd, return_all=True, use_clip=use_clip)

            # save
            x_t = x_t.detach().clone()
            if keep_all_samples or i == n_ts-1:
                x = x_t.detach().clone().to('cpu')#.permute(0, 2, 3, 1).reshape(*xsz)
                xs.append(x[None])
            else:
                pass
            tm.update(1)

    # return
    model.train()
    if keep_all_samples:
        return torch.cat(xs, dim=0).detach().clone()
    else:
        return xs[-1].detach().clone()
