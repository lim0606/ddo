import numpy as np
import torch


class FixedClassesSineData:
    def __init__(self,
                 batch_size,
                 num_points,
                 num_classes=50000,
                 seed=0,
                 rand_v=True,
                 max_iter=50000,
                 ):
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_classes = num_classes
        g = torch.Generator()
        g.manual_seed(seed)
        # self.amplitudes = torch.rand(num_classes, generator=g)*2. - 1. #np.linspace(-1., 1., num=num_classes)
        self.amplitudes = torch.rand(num_classes, generator=g)
        self.shifts = torch.linspace(-2., 2., steps=num_classes)
        self.rand_v = rand_v
        self.max_iter = max_iter

    def set_num_points(self, num_points):
        self.num_points = num_points

    def sample(self, num_samples, num_points=100, label=None, **kwargs):
        if label is None:
            label = torch.randint(high=len(self.amplitudes), size=(num_samples,))
        v_dim = x_dim = 1  # v and x dim are fixed for this dataset.

        # Generate data

        # Sample random amplitude
        a = torch.index_select(self.amplitudes, 0, label).view(num_samples, 1, 1)
        # Sample random shift
        b = torch.index_select(self.shifts, 0, label).view(num_samples, 1, 1)
        # Shape (num_points, v_dim)
        if self.rand_v:
            v = torch.rand(num_samples, num_points, v_dim) * 2. * np.pi - np.pi
        else:
            v = torch.linspace(-np.pi, np.pi, num_points).reshape(1, num_points, 1).repeat(num_samples, 1, v_dim) # bsz x num_points x v_dim

        # Shape (num_points, x_dim)
        x = a * torch.sin(v - b) # bsz x num_points x x_dim

        return (x, v), label

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter <= self.max_iter:
            self.iter += 1
            return self.sample(self.batch_size, self.num_points)
        else:
            raise StopIteration


from lib.gp import GaussianProcess
class MixturesOfGaussianProcesses:
    def __init__(self,
                 batch_size,
                 num_points,
                 seed=0,
                 rand_v=True,
                 max_iter=10000,
                 device="cpu",
                 ):
        self.batch_size = batch_size
        self.num_points = num_points
        g = torch.Generator()
        g.manual_seed(seed)

        self.a = torch.Tensor([10., -10.])
        self.b = torch.Tensor([-5., 5.])
        self.gp = GaussianProcess(xdim=1, vdim=1, exponent=2.0, vscale=0.1, sigma=np.sqrt(0.4))

        self.rand_v = rand_v
        self.max_iter = max_iter
        self.device = device

    def set_num_points(self, num_points):
        self.num_points = num_points

    def sample(self, num_samples, num_points=100, label=None, rand_v=None, **kwargs):
        if rand_v is None:
            rand_v = self.rand_v

        device = self.device
        if self.a.device != device:
            self.a = self.a.to(device)
            self.b = self.b.to(device)
            self.gp = self.gp.to(device)

        if label is None:
            label = torch.randint(high=2, size=(num_samples,), device=device)
        v_dim = x_dim = 1  # v and x dim are fixed for this dataset.

        if rand_v:
            v = torch.rand(num_samples, num_points, v_dim, device=device)
        else:
            v = torch.linspace(0., 1., num_points).reshape(1, num_points, 1).repeat(num_samples, 1, v_dim).to(device) # bsz x num_points x v_dim
        _, _, _, covmat_tril, _ = self.gp.get_covmat(v)
        noise = self.gp.sample(covmat_tril, return_noise=False)

        a = torch.index_select(self.a, 0, label).view(num_samples, 1, 1)
        b = torch.index_select(self.b, 0, label).view(num_samples, 1, 1)
        mean = a * v + b # bsz x num_points x x_dim

        x = mean + noise
        return (x, v), label

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter <= self.max_iter:
            self.iter += 1
            return self.sample(self.batch_size, self.num_points)
        else:
            raise StopIteration
