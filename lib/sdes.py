import numpy as np
import torch
import torch.nn as nn
import tqdm

from lib.dct import LinearDCT
from lib.ggp import RegularGridGaussianProcess
from lib.diffusion import get_frequency_scaling


class VariancePreservingSDE(nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self,
                 dim,
                 beta_min=0.1,
                 beta_max=20.0,
                 T=torch.Tensor([1.]),
                 t_epsilon=0.001,
                 disp_method:str='const',
                 gp_kwargs:dict={
                     'gp_type': 'independent',
                 },
                 **kwargs,
                ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.register_buffer('T', T)
        self.t_epsilon = t_epsilon
        self.dim = dim

        self.all_kwargs = kwargs
        assert disp_method == 'const'

        self.disp_method = disp_method
        self.dct = LinearDCT(dim=dim, dct_type='dct')
        self.idct = LinearDCT(dim=dim, dct_type='idct')
        self.freq_scaling = None #get_frequency_scaling(t, img_dim, dim=self.dim, self.disp_method, **self.all_kwargs)

        self.gp_kwargs = gp_kwargs
        self.gp_kwargs['dim'] = self.dim
        # self.gp = GaussianRF(**self.gp_kwargs)
        self.gp = RegularGridGaussianProcess(**self.gp_kwargs)

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    def alpha(self, *args, **kwargs):
        return self.mean_weight(*args, **kwargs)

    def var(self, t, **kwargs):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def std(self, t, **kwargs):
        return self.var(t) ** 0.5

    def f(self, t, x, **kwargs):
        if x is None:
            return - 0.5 * self.beta(t)
        else:
            return - 0.5 * self.beta(t) * x

    def g(self, t, x, **kwargs):
        return torch.sqrt(self.beta(t))

    def g2(self, t, x, **kwargs):
        return self.beta(t)

    def get_freq_scaling(self, t, img_dim):
        assert self.disp_method == 'const'
        if self.freq_scaling is None \
           or self.freq_scaling.shape[-1] != img_dim \
           or self.freq_scaling.shape[0] != len(t):
            self.freq_scaling = get_frequency_scaling(t, img_dim, dim=self.dim, method=self.disp_method, **self.all_kwargs)
        return self.freq_scaling

    def get_alpha_sigma(self, t, img_dim):
        # init
        t = t.view(-1)
        freq_scaling = self.get_freq_scaling(t, img_dim)
        a, sigma = self.alpha(t), self.std(t)
        alpha = expand_dim(a, self.dim) * freq_scaling
        sigma = expand_dim(sigma, self.dim)
        # if self.dim == 1:
        #     alpha = a[:, None, None] * freq_scaling # Combine dissipation and scaling.
        #     return alpha, sigma[:, None, None]
        # elif self.dim == 2:
        #     alpha = a[:, None, None, None] * freq_scaling # Combine dissipation and scaling.
        #     return alpha, sigma[:, None, None, None]
        # else:
        #     raise NotImplementedError
        return alpha, sigma

    def dct_blur_nd(self, x, freq_scaling):
        if self.dim == 1:
            assert x.dim() == 3 # B, C, H
            assert freq_scaling.dim() == 3
        elif self.dim == 2:
            assert x.dim() == 4 # B, C, H, W
            assert freq_scaling.dim() == 4
        else:
            raise NotImplementedError
        x_freq = self.dct(x)
        return self.idct(freq_scaling * x_freq)

    def diffuse(self, x, t, return_noise=False, **kwargs):
        """
        sample xt | x0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        # init
        t = t.view(-1)
        if self.dim == 1:
            assert x.dim() == 3 # B, C, H
        elif self.dim == 2:
            assert x.dim() == 4 # B, C, H, W
        else:
            raise NotImplementedError

        # get alpha and sigma
        alpha, sigma = self.get_alpha_sigma(t, x.shape[-1])

        # mean
        mu_diffuse = self.dct_blur_nd(x, alpha)

        # noise
        eps = self.gp.sample(x.shape[0], x.shape[-1], ch=x.shape[1]).to(x.device)

        z_t = mu_diffuse + sigma * eps

        return z_t, eps, mu_diffuse, sigma

    def sample(self, t, x0, covmat_tril=None, **kwargs):
        """
        sample xt | x0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        xt, epsilon, mu, sigma = self.diffuse(x=x0, t=t, covmat_tril=covmat_tril, **kwargs)
        return xt, epsilon, mu, sigma


class PluginReverseSDE(nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, model, model_type="epsilon", vtype="rademacher", debias=False):
        super().__init__()
        self.base_sde = base_sde
        assert model_type in ["score", "epsilon"]
        self.model_type = model_type
        self.model = model
        self.vtype = vtype
        self.debias = debias

    @property
    def dim(self):
        return self.base_sde.dim

    @property
    def T(self):
        return self.base_sde.T

    @property
    def gp(self):
        return self.base_sde.gp

    def diffuse(self, *args, **kwargs):
        return self.base_sde.diffuse(*args, **kwargs)

    def get_freq_scaling(self, *args, **kwargs):
        return self.base_sde.get_freq_scaling(*args, **kwargs)

    def dct_blur_nd(self, *args, **kwargs):
        return self.base_sde.dct_blur_nd(*args, **kwargs)

    def get_blur(self, t, x):
        # init
        t = t.view(-1)
        freq_scaling = self.get_freq_scaling(t, x.shape[-1])
        x_blurred = self.dct_blur_nd(x, freq_scaling)
        return x_blurred

    # drift
    def f(self, t, x, v, lmbd=0., **kwargs):
        _f = expand_dim(self.base_sde.f(self.T - t, None), self.dim)
        _g2 = expand_dim(self.base_sde.g2(self.T - t, None), self.dim)
        score = self.score(x, self.T - t, v)
        #return (1. - 0.5 * lmbd) * _g2 * score - self.base_sde.f(self.T - t, x)
        return (1. - 0.5 * lmbd) * _g2 * score - _f * self.get_blur(self.T - t, x)

    # diffusion
    def g(self, t, x, lmbd=0., **kwargs):
        g = expand_dim(self.base_sde.g(self.T - t, None), self.dim)
        return (1. - lmbd) ** 0.5 * g

    # epsilon
    def epsilon(self, y, s, v=None, **kwargs):
        if self.model_type == 'epsilon':
            return self.model(x=y, temp=s.view(-1), v=v)
        else:
            raise NotImplementedError

    # covmat * score
    def score(self, y, s, v=None, **kwargs):
        if self.model_type == 'epsilon':
            _std = expand_dim(self.base_sde.std(s, covmat_tril=None), self.dim)
            return - self.model(x=y, temp=s.view(-1), v=v) / _std
        else:
            raise ValueError

    @torch.enable_grad()
    def dsm(self, z0, v=None, covmat_tril=None, **kwargs):
        """
        denoising score matching loss
        """
        s_ = torch.rand(z0.size(0), device=z0.device).to() * self.T
        zt, target, _, _ = self.base_sde.sample(s_, z0, covmat_tril=covmat_tril)
        pred = self.epsilon(y=zt, s=s_, v=v, covmat_tril=covmat_tril)
        return ((pred - target) ** 2).view(z0.size(0), -1).sum(1, keepdim=False) / 2

    def denoise_step(self, x_t, t, delta, v=None, covmat_tril=None, lmbd=0., return_all=False, **kwargs):
        t = t.view(-1)

        # mu and sigma
        f = self.f(x=x_t, t=t, v=v, covmat_tril=covmat_tril, lmbd=lmbd)
        g = self.g(x=x_t, t=t, covmat_tril=covmat_tril, lmbd=lmbd)

        eps = self.gp.sample(x_t.shape[0], x_t.shape[-1], ch=x_t.shape[1]).to(x_t.device)

        mu_denoise = x_t + delta * f
        sigma_eps_denoise = delta ** 0.5 * g * eps # one step update of Euler Maruyama method with a step size delta

        # Sample from the denoising distribution.
        x_tm1 = mu_denoise + sigma_eps_denoise

        if return_all:
            return x_tm1, eps, mu_denoise, sigma_eps_denoise
        else:
            return x_tm1


@torch.no_grad()
def euler_maruyama_method(sde, num_steps, x_0, v=None, covmat_tril=None, lmbd=0., s_min=0.001, keep_all_samples=False, disable_tqdm=True):
    """
    Euler Maruyama method with a step size delta
    """
    # init
    sde.eval()
    device = sde.T.device
    T_ = sde.T.item()
    delta = (T_-s_min) / num_steps
    ts = np.linspace(s_min, T_-s_min, num_steps)
    ONES = torch.ones(batch_size,).cuda()

    # sample
    x_t = x_0.detach().clone().cuda()
    if keep_all_samples:
        xs = [x_t.detach().clone().to('cpu')[None]]
    else:
        xs = []

    with torch.no_grad() as _ , tqdm.tqdm(disable=disable_tqdm, position=0, leave=False, unit='steps', total=num_steps, desc='Euler-Maruyama method', ncols=80) as tm:
        for i in range(num_steps):
            # init t
            t = ts[i] * ONES

            if i == num_steps-1:
                _, _, mu_t, _ = sde.denoise_step(x_t=x_t, t=t, delta=delta, v=v, lmbd=lmbd, return_all=True)
                x_t = mu_t
            else:
                x_t, _, _, _ = sde.denoise_step(x_t=x_t, t=t, delta=delta, v=v, lmbd=lmbd, return_all=True)

            # save
            x_t = x_t.detach().clone()
            if keep_all_samples or i == num_steps-1:
                x = x_t.detach().clone().to('cpu')#.permute(0, 2, 3, 1).reshape(*xsz)
                xs.append(x[None])
            else:
                pass
            tm.update(1)

    # return
    sde.train()
    if keep_all_samples:
        return torch.cat(xs, dim=0).detach().clone()
    else:
        return xs[-1].detach().clone()


denoise = euler_maruyama_method
