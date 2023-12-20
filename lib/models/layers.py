import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.unet import conv1d, conv2d
from lib.models.unet import get_sinusoidal_positional_embedding
from lib.models.aliasfree_dct import Upsample1d, Upsample2d, Downsample1d, Downsample2d


def parse_kwargs(prefix, keep_prefix=False, **kwargs):
    new_kwargs = {}
    for key, val in kwargs.items():
        if key.startswith(prefix):
            if keep_prefix:
                new_key = key
            else:
                new_key = key.split(prefix,1)[-1]
            new_kwargs.update({new_key: val})
    return new_kwargs


def normalize(norm, out_channels=None):
    if norm is None:
        layer = nn.Identity()
    elif norm == 'instance_norm':
        layer = nn.InstanceNorm1d(num_features=out_channels)
    elif norm == 'layer_norm':
        layer = nn.LayerNorm()
    else:
        raise ValueError(f'Got {norm=} but expected None or layer_norm')
    return layer


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, n_dim=2, activation=nn.GELU, dropout=0., **kwargs):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.w1 = nn.Parameter(torch.ones(hidden_features, in_features, *(1,)*n_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_features))

        self.w2 = nn.Parameter(torch.ones(out_features, hidden_features, *(1,)*n_dim))
        self.b2 = nn.Parameter(torch.zeros(out_features))

        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # map order to actual conv, e.g. 2 -> F.conv2d, etc
        self.functional_conv = getattr(F, f'conv{n_dim}d')

        # init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w1, std=.02)
        nn.init.constant_(self.b1, 0.)
        nn.init.normal_(self.w2, std=.02)
        nn.init.constant_(self.b2, 0.)

    def forward(self, x):
        x = self.functional_conv(x, self.w1, bias=self.b1)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.functional_conv(x, self.w2, bias=self.b2)
        x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        return (f'(features): [{self.in_features}, {self.hidden_features}, {self.out_features}]'
                )


class PositionalEmbedding1d(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, pos_dim=1, act=nn.SiLU()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        self.main = nn.Sequential(
            conv1d(embedding_dim*pos_dim, hidden_dim, kernel_size=1, padding=0),
            act,
            conv1d(hidden_dim, output_dim, kernel_size=1, padding=0),
        )

    def _get_sinusoidal_positional_embedding(self, temp):
        """
        temp: bsz x t_dim x height
        """
        batch_size, t_dim, height = temp.shape
        temp = temp.permute(0, 2, 1).reshape(batch_size*height, t_dim)
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        return temb.reshape(batch_size, height, -1).permute(0, 2, 1).contiguous()

    def forward(self, temp):
        """
        temp: bsz x t_dim x height
        """
        temb = self._get_sinusoidal_positional_embedding(temp)
        temb = self.main(temb)
        return temb


class PositionalEmbedding2d(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, pos_dim=2, act=nn.SiLU()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        self.main = nn.Sequential(
            conv2d(embedding_dim*pos_dim, hidden_dim, kernel_size=1, padding=0),
            act,
            conv2d(hidden_dim, output_dim, kernel_size=1, padding=0),
        )

    def _get_sinusoidal_positional_embedding(self, temp):
        """
        temp: bsz x t_dim x height x width
        """
        batch_size, t_dim, height, width = temp.shape
        temp = temp.permute(0, 2, 3, 1).reshape(batch_size*height*width, t_dim)
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        return temb.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, temp):
        """
        temp: bsz x t_dim x height
        """
        temb = self._get_sinusoidal_positional_embedding(temp)
        temb = self.main(temb)
        return temb


class PointwiseOp(nn.Module):
    def __init__(self, in_channels, out_channels, dim, scale_factor=1, stopband_rel=2**0.3, filter_size=9, stride=1, bias=True):
        super().__init__()

        if dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        else:
            raise NotImplementedError

        if scale_factor > 1:
            if dim == 1:
                self.scale = Upsample1d(scale_factor=scale_factor, stopband_rel=stopband_rel, filter_size=filter_size)
            elif dim == 2:
                self.scale = Upsample2d(scale_factor=scale_factor, stopband_rel=stopband_rel, filter_size=filter_size)
            else:
                raise NotImplementedError
        elif scale_factor < 1:
            if dim == 1:
                self.scale = Downsample1d(scale_factor=scale_factor, stopband_rel=stopband_rel, filter_size=filter_size)
            elif dim == 2:
                self.scale = Downsample2d(scale_factor=scale_factor, stopband_rel=stopband_rel, filter_size=filter_size)
            else:
                raise NotImplementedError
        else:
            self.scale = None

    def forward(self, x):
        if self.scale is not None:
            x = self.scale(x)
        return self.conv(x)

