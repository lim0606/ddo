import torch
import torch.fft as fft

import math

#Gaussian random fields with Matern-type covariance: C = sigma^2 (-Lap + tau^2 I)^-alpha
#Generates random field samples on the domain [0,L1] x [0,L2]

class PeriodicGaussianRF2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, device=None, dtype=torch.float32):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - 2.0))

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