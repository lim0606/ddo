import math
import functools
import numpy as np

import torch
import torch.fft as fft

#Gaussian random fields with Matern-type covariance: C = sigma^2 (-Lap + tau^2 I)^-alpha
#Generates random field samples on the domain [0,L]


class IndependentGaussianRF1d(object):

    def __init__(self, s1:int, mean=None, device=None, dtype=torch.float32, **kwargs):
        """
        s: resolution
        """
        self.s1 = s1
        self.mean = mean
        self.dtype = dtype
        self.device = device

    def sample(self, N, **kwargs):
        return torch.randn(N, self.s1, dtype=self.dtype, device=self.device)

    def __repr__(self):
        return f'IndependentGaussianRF1d(s1={self.s1})'


class IndependentGaussianRF2d(object):

    def __init__(self, s1:int, s2:int, mean=None, device=None, dtype=torch.float32, **kwargs):
        """
        s1: height
        s2: width
        """
        self.s1 = s1
        self.s2 = s2
        self.mean = mean
        self.dtype = dtype
        self.device = device

    def sample(self, N, **kwargs):
        return torch.randn(N, self.s1, self.s2, dtype=self.dtype, device=self.device)

    def __repr__(self):
        return f'IndependentGaussianRF2d(s1=self.s1)'


#Dirichlet and Neumann boundaries
class GaussianRF1d(object):

    def __init__(self, s1:int, L:float=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="dirichlet", device=None, dtype=torch.float32, **kwargs):
        """
        s: resolution
        L: maximum value of x \in [0, L]
        alpha: smoothness
        tau: length scale
        sigma: magnitude
        """
        self.s1 = s = s1
        self.L = L

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - 1.0))

        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma
        self.boundary = boundary

        const = ((math.pi**2))/(L**2)
        norm_const = math.sqrt(2.0/L)

        k = torch.arange(start=1, end=s//2 + 1, step=1).view(s//2, 1).to(self.device)
        sqrt_eig = sigma*((const*k**2 + tau**2)**(-alpha/2.0))
        x = (math.pi/L)*k*torch.linspace(0, L, s, dtype=self.dtype, device=self.device).repeat(s//2, 1)

        if boundary == 'dirichlet':
            self.phi = norm_const*sqrt_eig*x.sin()
        elif boundary == 'neumann':
            self.phi = norm_const*sqrt_eig*x.cos()
        else:
            raise ValueError(f'Boundary type \'{boundary}\' is not implemented.')

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1//2, dtype=self.dtype, device=self.device)

        u = torch.mm(xi, self.phi)

        if self.mean is not None:
            u += self.mean

        return u

    def __repr__(self):
        return f'GaussianRF1d( \n' \
               f'    L={self.L},\n' \
               f'    alpha={self.alpha},\n' \
               f'    tau={self.tau},\n' \
               f'    sigma={self.sigma},\n' \
               f'    boundary={self.boundary},\n' \
               f')'


#Periodic boundary
class PeriodicGaussianRF1d(object):

    def __init__(self, s1, L=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, device=None, dtype=torch.float32, **kwargs):
        """
        s: resolution
        L: maximum value of x \in [0, L]
        alpha: smoothness
        tau: length scale
        sigma: magnitude
        """

        self.s1 = s = s1
        self.L = L

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - 1.0))

        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

        const = (4*(math.pi**2))/(L**2)
        norm_const = math.sqrt(2.0/L)

        k = torch.cat((torch.arange(start=0, end=s//2, step=1),\
                       torch.arange(start=-s//2, end=0, step=1)), 0).type(dtype).to(device)

        self.sqrt_eig = s*sigma*norm_const*((const*k**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0] = 0.0
        self.sqrt_eig[s//2:] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, 2, dtype=self.dtype, device=self.device)

        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]

        u = fft.ifft(torch.view_as_complex(xi), n=self.s1).imag

        if self.mean is not None:
            u += self.mean

        return u

    def __repr__(self):
        return f'PeriodicGaussianRF1d( \n' \
               f'    L={self.L},\n' \
               f'    alpha={self.alpha},\n' \
               f'    tau={self.tau},\n' \
               f'    sigma={self.sigma},\n' \
               f')'


class PeriodicGaussianRF2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, device=None, dtype=torch.float32, **kwargs):

        self.s1 = s1
        self.s2 = s2
        self.L1 = L1
        self.L2 = L2
        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - 2.0))

        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)
        norm_const = math.sqrt(2.0/(L1*L2))

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2).type(dtype).to(device)

        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1),\
                                torch.arange(start=-s2//2, end=0, step=1)), 0)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*sigma*norm_const*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0
        self.sqrt_eig[torch.logical_and(k1 + k2 <= 0.0, torch.logical_or(k1 + k2 != 0.0, k1 <= 0.0))] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2, 2, dtype=self.dtype, device=self.device)

        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]

        u = fft.ifft2(torch.view_as_complex(xi), s=(self.s1, self.s2)).imag

        if self.mean is not None:
            u += self.mean

        return u

    def __repr__(self):
        return f'PeriodicGaussianRF2d( \n' \
               f'    (L1, L2)=({self.L1}, {self.L2}),\n' \
               f'    alpha={self.alpha},\n' \
               f'    tau={self.tau},\n' \
               f'    sigma={self.sigma},\n' \
               f')'


class GaussianRF(object):

    def __init__(self, dim:int, gp_type:str, ch:int=1, **kwargs):
        assert gp_type in ['independent', 'periodic', 'non-periodic']
        self.dim = dim
        self.ch = ch
        self.gp_type = gp_type
        assert 's1' not in kwargs

        if dim == 1:
            if self.gp_type == 'independent':
                sampler = IndependentGaussianRF1d
            elif self.gp_type == 'periodic':
                sampler = PeriodicGaussianRF1d
            elif self.gp_type == 'non-periodic':
                sampler = GaussianRF1d
            else:
                raise ValueError
        elif dim == 2:
            if self.gp_type == 'independent':
                sampler = IndependentGaussianRF2d
            elif self.gp_type == 'periodic':
                sampler = PeriodicGaussianRF2d
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.init_sampler = functools.partial(sampler, **kwargs)
        self.sampler = self.init_sampler(s1=32, s2=32)

    @property
    def device(self):
        return self.sampler.sqrt_eig.device

    def sample(self, N, img_height, ch=None, **kwargs):
        if ch is None:
            ch = self.ch
        if self.sampler is None or self.sampler.s1 != img_height:
            self.sampler = self.init_sampler(s1=img_height, s2=img_height)
        u = self.sampler.sample(N*ch, **kwargs)
        usz = list(u.shape)
        new_usz = [N, ch] + usz[1:]
        return u.reshape(*new_usz)

    def __repr__(self):
        return self.sampler.__repr__()
