import os
import functools
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process

from thirdparty.random_fields import *
from utils import utils
from utils import datasets
from utils.ema import EMA
from utils.utils import save_checkpoint, load_checkpoint
from utils.visualize import get_grid_image
from utils.evaluate import compute_feature_stats_for_dataset, compute_feature_stats_for_generator
from utils.evaluate import calculate_frechet_distance, calculate_precision_recall
from metrics.image import metric_utils

from lib.models.aliasfree_dct import Upsample2d, Downsample2d


#######################################################################
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

def to_01(x):
    return (x + 1.) / 2.

def to_01_clip(x):
    x = (x + 1.) / 2.
    return torch.clip(x, 0, 1)

def from_sdf_to_01(x):
    return (-1.*x + 1.) / 2.

def from_sdf_to_01_clip(x):
    x = (-1.*x + 1.) / 2.
    return torch.clip(x, 0, 1)

def from_sdf_to_mask(x):
    return (x < 0.).float()

def x_to_image(x):
    batch_size, num_points, _ = x.shape
    img_height = int(np.sqrt(num_points))
    assert img_height*img_height == num_points
    return x.reshape(batch_size, img_height, img_height, -1).permute(0, 3, 1, 2).contiguous()

def image_to_x(x):
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])

def xv_to_mask(x):
    x, _ = xv_to_image(x)
    return from_sdf_to_mask(x)

def x_to_mask(x):
    x = x_to_image(x)
    return from_sdf_to_mask(x)



#######################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2, modes1 = None, modes2 = None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1 #output dimensions
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 #if not given take the highest number of modes can be taken
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2))
        return x

class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel,dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

class Generator(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, half_modes, pad = 0, factor = 3/4):
        super(Generator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, half_modes, half_modes)

        self.conv1 = SpectralConv2d(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, half_modes//2, half_modes//2)

        self.conv2 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16, half_modes//4, half_modes//4)

        self.conv2_1 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8, half_modes//8, half_modes//8)

        self.conv2_9 = SpectralConv2d(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16, half_modes//8, half_modes//8)


        self.conv3 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, half_modes//4, half_modes//4)

        self.conv4 = SpectralConv2d(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, half_modes//2, half_modes//2)

        self.conv5 = SpectralConv2d(4*factor*self.d_co_domain, self.d_co_domain, 64, 64, half_modes, half_modes) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain, 48, 48) #

        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #

        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16) #

        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8)

        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16)

        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #

        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48)

        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain, 64, 64) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, 1)

    def forward(self, x):

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)


        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])

        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]


        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)

        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)

        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)


        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x_c5 = x_c5.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)

        x_out = torch.tanh(x_out)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def kernel(in_chan=2, up_dim=32):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, 1, bias=False)
            )
    return layers

class Discriminator(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, half_modes, kernel_dim=16, pad = 0, factor = 3/4):
        super(Discriminator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic
        self.kernel_dim = kernel_dim
        self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, half_modes, half_modes)

        self.conv1 = SpectralConv2d(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, half_modes//2, half_modes//2)

        self.conv2 = SpectralConv2d(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16, half_modes//4, half_modes//4)

        self.conv2_1 = SpectralConv2d(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8, half_modes//8, half_modes//8)

        self.conv2_9 = SpectralConv2d(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16, half_modes//8, half_modes//8)


        self.conv3 = SpectralConv2d(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32, half_modes//4, half_modes//4)

        self.conv4 = SpectralConv2d(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48, half_modes//2, half_modes//2)

        self.conv5 = SpectralConv2d(4*factor*self.d_co_domain, self.d_co_domain, 64, 64, half_modes, half_modes) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain,48, 48) #

        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #

        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16) #

        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, 8, 8)

        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, 16, 16)

        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, 32, 32) #

        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, 48, 48)

        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain, 64, 64) # will be reshaped

        self.fc1 = nn.Linear(2*self.d_co_domain, 4*self.d_co_domain)
        self.fc2 = nn.Linear(4*self.d_co_domain, 1)

        # kernel for last functional operation

        self.knet = kernel(2, self.kernel_dim)


    def forward(self, x):
        batch_size = x.shape[0]
        res1 = x.shape[1]
        res2 = x.shape[2]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)


        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])

        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]


        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)

        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)

        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)


        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x = x_c5
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        kx = self.knet(grid)

        kx = kx.view(batch_size, -1, 1)

        x = x.view(batch_size, -1, 1)
        x = torch.einsum('bik,bik->bk', kx, x)/(res1*res2)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for GAN"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1.0/real_images.shape[1]) ** 2)
    return gradient_penalty

@torch.no_grad()
def sample_image(batch_size, gen, grf, transform=None, clip=True, **kwargs):
    z = grf.sample(batch_size).unsqueeze(-1)
    x_syn = gen(z)

    if transform is not None:
        x_syn = transform(x_syn)

    if clip:
        x_syn = torch.clip(x_syn, 0, 1)

    return x_syn

def create_generator(num_samples, batch_size, gen, grf, transform=None, clip=False):
    num_iters = int(np.ceil(num_samples / batch_size))
    for i in range(num_iters):
        x_syn = sample_image(batch_size, gen, grf, transform=transform, clip=clip)
        yield x_syn.float().permute(0, 3, 1, 2).contiguous()



#######################################################################
# train
#######################################################################

def save_checkpoint(args, global_step, best_fid_score,
                    gen, gen_optimizer,
                    disc, disc_optimizer,
                    checkpoint_name='checkpoint.pt',
                   ):
    if args.global_rank == 0:
        content = {'global_step': global_step,
                   'args': args,
                   'best_fid_score': best_fid_score,
                   'gen_state_dict': gen.state_dict(),
                   'gen_optimizer': gen_optimizer.state_dict(),
                   'disc_state_dict': disc.state_dict(),
                   'disc_optimizer': disc_optimizer.state_dict(),
                   }
        checkpoint_file = os.path.join(args.folder_path, checkpoint_name)
        torch.save(content, checkpoint_file)
    return

def load_checkpoint(checkpoint_file, gen, gen_optimizer, disc, disc_optimizer):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    global_step = checkpoint['global_step']
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])

    disc.load_state_dict(checkpoint['disc_state_dict'])
    disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

    best_fid_score = checkpoint['best_fid_score']
    return global_step, best_fid_score

def save_model(args, global_step, gen, best_fid_score, checkpoint_name='checkpoint_fid.pt'):
    assert checkpoint_name in {'checkpoint_init.pt', 'checkpoint_fid.pt'}
    if args.global_rank == 0:
        # Because we already swapped EMA parameters, we only save the gen models here.
        content = {'global_step': global_step,
                   'args': args,
                   'best_fid_score': best_fid_score,
                   'gen_state_dict': gen.state_dict(),
                   }
        checkpoint_file = os.path.join(args.folder_path, checkpoint_name)
        torch.save(content, checkpoint_file)
    return

def load_model(checkpoint_file, distributed=None):
    # load model
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    count = checkpoint['global_step']
    args = checkpoint['args']
    _, gen = get_model(args)
    gen = gen.cuda()
    if distributed is None:
        distributed = args.distributed
    utils.broadcast_params(gen.parameters(), distributed)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    best_fid_score = checkpoint['best_fid_score']
    print('loaded the model at iter {:d}'.format(count))
    return gen, count, best_fid_score, args

def get_model(args):
    D = Discriminator(
        args.input_dim+args.coord_dim,
        args.d_co_domain,
        half_modes=args.modes//2,
        pad=args.npad,
        factor=1.,
    ).cuda()
    G = Generator(
        args.input_dim+args.coord_dim,
        args.d_co_domain,
        half_modes=args.modes//2,
        pad=args.npad,
        factor=1.,
    ).cuda()
    return D, G

def init_model(args, resume:bool=None):
    # init model
    disc, gen = get_model(args)

    # broadcast params
    utils.broadcast_params(disc.parameters(), args.distributed)
    utils.broadcast_params(gen.parameters(), args.distributed)

    # init optimizer
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    gen_optimizer  = torch.optim.Adam(gen.parameters(),  lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # add ema
    disc_optimizer = EMA(disc_optimizer, ema_decay=args.ema_decay)
    gen_optimizer  = EMA(gen_optimizer,  ema_decay=args.ema_decay)

    # resume
    if hasattr(args, 'checkpoint_file'):
        checkpoint_file = os.path.join(args.folder_path, args.checkpoint_file)
    else:
        checkpoint_file = os.path.join(args.folder_path, 'checkpoint.pt')
    resume = args.resume if resume is None else resume
    if resume and os.path.exists(checkpoint_file):
        print('loading the model.')
        count, best_fid_score = load_checkpoint(checkpoint_file, gen, gen_optimizer, disc, disc_optimizer)
        print('loaded the model at iter {:d}'.format(count))
    else:
        count = 0
        best_fid_score = 1e10

    return gen, gen_optimizer, disc, disc_optimizer, count, best_fid_score


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
    _, val_loader, _ = datasets.get_loaders_eval(
        dataset=args.dataset,
        root=args.data,
        distributed=args.distributed,
        batch_size=args.vis_batch_size,
        centered=False,
    )

    # init model
    (gen,
     gen_optimizer,
     disc,
     disc_optimizer,
     count,
     best_fid_score,
     ) = init_model(args)
    logging.info(disc)
    logging.info(gen)

    # init loss
    fn_loss = nn.BCEWithLogitsLoss()

    # upsample
    if args.upsample and args.train_img_height < args.upsample_resolution:
        # init upsample
        scale_factor = int(args.upsample_resolution/args.train_img_height)
        assert scale_factor*args.train_img_height == args.upsample_resolution
        upsample = Upsample2d(
            # use_fft=True,
            scale_factor=scale_factor,
            filter_size=args.upsample_filter_size,
            use_radial=args.use_radial,
        ).cuda()

        # init grf
        grf = GaussianRF_idct(args.coord_dim, args.upsample_resolution, alpha=1.5, tau=1.0, device='cuda')
    else:
        # init grf
        grf = GaussianRF_idct(args.coord_dim, args.train_img_height, alpha=1.5, tau=1.0, device='cuda')

    # init model generator
    fn_create_generator = functools.partial(
            create_generator,
            gen=gen,
            grf=grf,
            transform=from_sdf_to_mask,
    )

    # train
    start_time = time.time()
    if count >= args.num_iterations:
        not_finished = False
        logging.info('Finished training at iter {:d}'.format(count))
    else:
        not_finished = True
    loss_G = torch.tensor(0)
    while not_finished:
        if args.distributed:
            train_loader.sampler.set_epoch(count + args.seed)
            val_loader.sampler.set_epoch(0)

        for j, (x, _) in enumerate(train_loader):
            # init batch
            x = x.cuda()

            # upsample
            if args.upsample:
                x = image_to_x(upsample(x_to_image(x)))
                x = x.reshape(x.shape[0], args.upsample_resolution, args.upsample_resolution, -1)#.permute(0, 3, 1, 2)
            else:
                x = x.reshape(x.shape[0], args.train_img_height, args.train_img_height, -1)#.permute(0, 3, 1, 2)

            disc_optimizer.zero_grad()

            z = grf.sample(x.shape[0]).unsqueeze(-1)
            x_syn = gen(z)

            loss_W = -torch.mean(disc(x)) + torch.mean(disc(x_syn.detach()))

            gradient_penalty = calculate_gradient_penalty(disc, x.data, x_syn.data, x.device)

            loss_D = loss_W + args.lmbd_grad * gradient_penalty

            loss_D.backward()
            utils.average_gradients(disc.parameters(), args.distributed)
            disc_optimizer.step()

            # Train G
            if (j + 1) % args.n_critic == 0:
                gen_optimizer.zero_grad()

                x_syn = gen(grf.sample(x.shape[0]).unsqueeze(-1))

                loss_G = -torch.mean(disc(x_syn))

                loss_G.backward()
                utils.average_gradients(gen.parameters(), args.distributed)
                gen_optimizer.step()

            # update
            count += 1

            ''' print '''
            if args.global_rank == 0 and count == 1 or count % args.print_every == 0:
                elapsed = time.time() - start_time # set log info
                logging.info('| iter:\t{:d} | loss_D:\t{:.2f} | loss_G:\t{:.2f}'.format(count, loss_D.item(), loss_G.item()))
                writer.add_scalar('train/loss_G/iter', loss_G.item(), count)
                writer.add_scalar('train/loss_D/iter', loss_D.item(), count)
                start_time = time.time() # reset log info

            if args.save_every > 0 and count % args.save_every == 0:
                save_checkpoint(args, count, best_fid_score,
                                gen, gen_optimizer,
                                disc, disc_optimizer,
                                checkpoint_name='checkpoint.pt',
                               )
                logging.info('Saved checkpoint at iter {:d}'.format(count))

            if args.ckpt_every > 0 and count % args.ckpt_every == 0:
                save_checkpoint(args, count, best_fid_score,
                                gen, gen_optimizer,
                                disc, disc_optimizer,
                                checkpoint_name='checkpoint_{}.pt'.format(count),
                               )
                logging.info('Saved checkpoint at iter {:d}'.format(count))

            if args.vis_every > 0 and count % args.vis_every == 0:
                # switch to EMA parameters
                if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema'):
                    gen_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                if args.global_rank == 0:
                    sample = sample_image(args.vis_batch_size, gen, grf, transform=from_sdf_to_01, clip=True).permute(0, 3, 1, 2)
                    nrow = int(args.vis_batch_size**0.5)
                    writer.add_image('train/samples', get_grid_image(sample[:nrow**2], nrow=nrow, pad_value=0, padding=2, to_numpy=False), count)
                    writer.flush()
                    logging.info('Visualized samples at iter {:d}'.format(count))

                if args.plot and args.global_rank == 0:
                    sample = sample_image(args.vis_batch_size, gen, grf, transform=None, clip=False)
                    matplotlib.use(MYBACKEND)
                    x_plot = sample.detach().cpu().reshape(sample.shape[0], -1, sample.shape[-1])
                    v_plot = gen.get_grid(sample.shape, sample.device)
                    v_plot = v_plot.cpu().reshape(v_plot.shape[0], -1, v_plot.shape[-1])
                    plot_contourf(x_plot[:min(args.vis_batch_size, 4)].cpu(),
                                  v_plot[:min(args.vis_batch_size, 4)].cpu(),
                                  img_height=sample.shape[1],
                                  nrows=1, ncols=4, figsize=(20,4))

                # switch back to original parameters
                if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema'):
                    gen_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            if args.eval_every > 0 and count % args.eval_every == 0:
                # switch to EMA parameters
                if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema'):
                    gen_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                # eval
                eval_start_time = time.time()
                val_fid_ema, _, _ = evaluate(args, fn_create_generator, count=count, eval_metric='fid')
                eval_elapsed = time.time() - eval_start_time
                logging.info('-'*60)
                logging.info('| iter {:6d} | {:5.2f} sec | fid {:5.3f}'
                      .format(
                      count,
                      eval_elapsed,
                      val_fid_ema,
                      ))
                logging.info('-'*60)
                writer.add_scalar('train/fid_ema/iter', val_fid_ema, count)
                writer.flush()

                # save best fid model
                if args.global_rank == 0 and val_fid_ema < best_fid_score:
                    best_fid_score = val_fid_ema
                    logging.info('Saving the model for FID.')
                    save_model(args, count, gen, best_fid_score, checkpoint_name='checkpoint_fid.pt')

                # switch back to original parameters
                if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema'):
                    gen_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            if count >= args.num_iterations:
                save_checkpoint(args, count, best_fid_score,
                                gen, gen_optimizer,
                                disc, disc_optimizer,
                                checkpoint_name='checkpoint.pt',
                               )
                not_finished = False
                logging.info('Finished training at iter {:d}'.format(count))
                break

    return gen, gen_optimizer, disc, disc_optimizer, count, None


def evaluate(args, create_generator, count, eval_metric='fid'):
    # get datasets
    dataset_name = 'mnistsdf_{:d}'.format(args.eval_img_height)
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
        batch_size=args.eval_batch_size_per_gpu,
        transform=xv_to_mask,
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
           img_height=args.eval_img_height,
        ),
    )
    gen_stats = compute_feature_stats_for_generator(
        gen_opts,
        create_generator,
        detector_url,
        detector_kwargs,
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
    #if args.global_rank == 0 and eval_metric == 'pr':
    if eval_metric == 'pr':
        #real_features = torch.from_numpy(real_features).cuda()
        #gen_features = torch.from_numpy(gen_features).cuda()
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

def test(args):
    logging, writer = utils.common_init(args.global_rank, args.seed, args.folder_path, suffix='test')
    logging.info(f'folder path: {args.folder_path}')
    logging.info(str(args))

    # init grf
    grf = GaussianRF_idct(args.coord_dim, args.eval_img_height, alpha=1.5, tau=1.0, device='cuda')

    # load model
    if hasattr(args, 'checkpoint_file') and os.path.basename(args.checkpoint_file) in {'checkpoint_init.pt', 'checkpoint_fid.pt'}:
        gen, count, best_fid_score, train_args = load_model(args.checkpoint_file, distributed=args.distributed)
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

        args = train_args
        gen_optimizer = None
        args.eval_use_ema = False
    else:
        gen, gen_optimizer, _, _, count, _ = init_model(args, resume=True)

    # init model generator
    fn_create_generator = functools.partial(
        create_generator,
        gen=gen,
        grf=grf,
        transform=from_sdf_to_mask,
    )

    # switch to EMA parameters
    if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema'):
        gen_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    # evaluate
    fid, precision, recall = -1., -1., -1.
    if args.eval_fid:
        eval_start_time = time.time()
        fid, _, _ = evaluate(args, fn_create_generator, count=count, eval_metric='fid')
        eval_elapsed = time.time() - eval_start_time
        logging.info('-'*60)
        logging.info('| iter {:6d} | {:5.2f} sec '.format(count, eval_elapsed))
        logging.info('| fid {:5.3f}'.format(fid))
        logging.info('-'*60)
        writer.add_scalar('test/fid{}/{:d}x{:d}/iter'.format(
                              '_ema' if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema') else '',
                              args.eval_img_height, args.eval_img_height),
                          fid,
                          count,
                         )
    if args.eval_pr:
        eval_start_time = time.time()
        _, precision, recall = evaluate(args, fn_create_generator, count=count, eval_metric='pr')
        eval_elapsed = time.time() - eval_start_time
        logging.info('-'*60)
        logging.info('| iter {:6d} | {:5.2f} sec '.format(count, eval_elapsed))
        logging.info('| prec {:5.3f}'.format(precision))
        logging.info('| rec {:5.3f}'.format(recall))
        logging.info('-'*60)
        writer.add_scalar('test/prec{}/{:d}x{:d}/iter'.format(
                              '_ema' if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema') else '',
                              args.eval_img_height, args.eval_img_height),
                          fid,
                          count,
                         )
        writer.add_scalar('test/recall{}/{:d}x{:d}/iter'.format(
                              '_ema' if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema') else '',
                              args.eval_img_height, args.eval_img_height),
                          fid,
                          count,
                         )
    writer.close()

    # switch back to original parameters
    if args.eval_use_ema and hasattr(gen_optimizer, 'swap_parameters_with_ema'):
        gen_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    return fid, precision, recall


_folder_name_key_tuples = [
    ('model', ''),
    ('modes', 'md'),
    ('d_co_domain', 'ch'),
    ('lmbd_grad', 'lmbd'),
    ('n_critic', 'nc'),
    ('train_img_height', 'nht'),
    ('upsample', 'up'),
    ('upsample_resolution', 'res'),
    ('use_radial', 'rf'),
    ('train_batch_size', 'b'),
    ('lr', 'lr'),
    ('weight_decay', 'wd'),
    ('ema_decay', 'ema'),
    ('seed', 'sd'),
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--command_type", type=str, default='train', choices=['train', 'test'])

    # i/o
    parser.add_argument('--exp_path', type=str, default='results')
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--ckpt_every', type=int, default=100000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--vis_every', type=int, default=10000)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)

    # dataset
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data', type=str, default='datasets', help='data root')

    # model
    parser.add_argument("--model", type=str, default='gano-uno')
    parser.add_argument("--npad", type=int, default=0)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--d_co_domain", type=int, default=32)
    parser.add_argument("--lmbd_grad", type=float, default=10.0)
    parser.add_argument("--n_critic", type=int, default=10)

    # optimization
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--vis_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay factor')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 of Adam optimizer')
    parser.add_argument('--num_iterations', type=int, default=1000000,
                        help='number of iterations for training')

    # train
    parser.add_argument('--train_img_height', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--coord_dim', type=int, default=2)
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument("--upsample", action='store_true', default=False)
    parser.add_argument('--upsample_resolution', type=int, default=64)
    parser.add_argument('--upsample_filter_size', type=int, default=9)
    parser.add_argument("--use_radial", action='store_true', default=False)

    # eval
    parser.add_argument('--eval_img_height', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='input batch size for evaluation')
    parser.add_argument('--eval_use_ema', action='store_true', default=True)
    parser.add_argument('--eval_fid', action='store_true', default=True)
    parser.add_argument('--eval_pr', action='store_true', default=False)
    parser.add_argument('--eval_num_samples', type=int, default=50000,
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
    parser.add_argument('--master_port', type=str, default='6020',
                        help='port for master')

    args = parser.parse_args()

    _folder_name = []
    for k, abbr in _folder_name_key_tuples:
        if not hasattr(args, k) or getattr(args, k) is None:
            continue
        if k.startswith('upsample_') and args.upsample is False:
            continue
        if k == 'ch_mult':
            _folder_name += [abbr+''.join([str(i) for i in getattr(args, k)])]
        elif k == 'act':
            _folder_name += [abbr+getattr(args, k).__class__.__name__.lower()]
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

    # run
    main = train if args.command_type == 'train' else 'test'
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
