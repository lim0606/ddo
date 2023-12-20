import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty import torch_dct


class LinearDCT(nn.Module):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, dim, dct_type, norm='ortho'):
        super().__init__()
        #self.height = N = height
        self.dim = dim
        self.dct_type = dct_type
        self.norm = norm
        self.register_buffer('weight', None, persistent=False)

    def initialize(self, height):
        # initialise using dct function
        I = torch.eye(height)
        if self.dct_type == 'dct1':
            self.weight = torch_dct.dct1(I).data.t()
        elif self.dct_type == 'idct1':
            self.weight = torch_dct.idct1(I).data.t()
        elif self.dct_type == 'dct':
            self.weight = torch_dct.dct(I, norm=self.norm).data.t()
        elif self.dct_type == 'idct':
            self.weight = torch_dct.idct(I, norm=self.norm).data.t()
        else:
            raise ValueError

    def forward(self, x):
        if self.weight is None or self.weight.shape[-1] != x.shape[-1]:
            self.initialize(x.shape[-1])
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)

        if self.dim == 1:
            return self.apply_linear_1d(x)
        elif self.dim == 2:
            return self.apply_linear_2d(x)
        elif self.dim == 3:
            return self.apply_linear_3d(x)
        else:
            raise ValueError

    def apply_linear_1d(self, x):
        """Can be used with a LinearDCT layer to do a 2D DCT.
        :param x: the input signal
        :param linear_layer: any PyTorch Linear layer
        :return: result of linear layer applied to last 2 dimensions
        """
        X1 = F.linear(x, self.weight, None)
        return X1

    def apply_linear_2d(self, x):
        """Can be used with a LinearDCT layer to do a 2D DCT.
        :param x: the input signal
        :param linear_layer: any PyTorch Linear layer
        :return: result of linear layer applied to last 2 dimensions
        """
        X1 = F.linear(x, self.weight, None)
        X2 = F.linear(X1.transpose(-1, -2), self.weight, None)
        return X2.transpose(-1, -2)

    def apply_linear_3d(self, x):
        """Can be used with a LinearDCT layer to do a 3D DCT.
        :param x: the input signal
        :param linear_layer: any PyTorch Linear layer
        :return: result of linear layer applied to last 3 dimensions
        """
        X1 = F.linear(x, self.weight, None)
        X2 = F.linear(X1.transpose(-1, -2), self.weight, None)
        X3 = F.linear(X2.transpose(-1, -3), self.weight, None)
        return X3.transpose(-1, -3).transpose(-1, -2)

    def extra_repr(self) -> str:
        return (f'(dct_type): {self.dct_type}\n'
                f'(dim): {self.dim}\n'
                f'(norm): {self.norm}'
                )


class LinearDCT1d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=1, dct_type='dct', norm=norm)

class LinearDCT2d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=2, dct_type='dct', norm=norm)

class LinearDCT3d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=3, dct_type='dct', norm=norm)


class LinearIDCT1d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=1, dct_type='idct', norm=norm)

class LinearIDCT2d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=2, dct_type='idct', norm=norm)

class LinearIDCT3d(LinearDCT):
    def __init__(self, norm='ortho'):
        super().__init__(dim=3, dct_type='idct', norm=norm)
