import argparse
import torch
import torch.nn as nn

from lib.models.layers import parse_kwargs, normalize, MLP
from lib.models.fourier import SpectralConv1d, SpectralConv2d


def skip_connection(in_features, out_features, n_dim=2, bias=False, type="soft-gating"):
    """
    Copied from https://github.com/neuraloperator/neuraloperator/blob/0.1.1/neuralop/models/skip_connections.py#L1C1-L36C1
    """
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D. 
    bias : bool, optional
        whether to use a bias, by default False
    type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if type.lower() == 'soft-gating':
        return SoftGating(in_features=in_features, out_features=out_features, bias=bias, n_dim=n_dim)
    elif type.lower() == 'linear':
        return getattr(nn, f'Conv{n_dim}d')(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
    elif type.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Got skip-connection {type=}, expected one of {'soft-gating', 'linear', 'id'}.")


class SpectralResidualBlock1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes_height,
                 temb_ch:int=None,
                 act=nn.GELU(),
                 use_mlp=False,
                 mlp:dict={'expansion': 1.0, 'dropout': 0},
                 norm=None,
                 skip='identity',
                 decomposition_kwargs=dict(),
                 use_pointwise_op=False,
                 cond_type="improved",
                 **kwargs,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.act = act
        assert cond_type in ["ddpm", "improved"]
        self.cond_type = cond_type

        if temb_ch is not None and temb_ch > 0:
            self.temb_proj = nn.Linear(temb_ch, out_channels if cond_type == "ddpm" else out_channels*2)
        else:
            self.temb_proj = None
        self.norm1 = norm(in_channels, modes_height=modes_height) if norm is not None else nn.Identity()
        self.conv1 = SpectralConv1d(
                in_channels, out_channels, modes_height,
                skip=use_pointwise_op,
                )
        self.norm2 = norm(out_channels, modes_height=modes_height) if norm is not None else nn.Identity()
        self.conv2 = SpectralConv1d(
                out_channels, out_channels, modes_height,
                skip=use_pointwise_op,
                )

        if in_channels != out_channels:
            skip = 'linear'
        self.shortcut = skip_connection(in_channels, out_channels, type=skip, n_dim=1)

        if use_mlp:
            mlp = argparse.Namespace(**mlp)
            self.mlp = MLP(
                in_features=out_channels,
                hidden_features=int(round(out_channels*mlp.expansion)),
                n_dim=1,
                dropout=mlp.dropout,
            )
            self.mlp_skip = skip_connection(out_channels, out_channels, type=skip, n_dim=1)
        else:
            self.mlp = None

    def forward(self, x, temb=None):
        # forward conv1
        x_res = self.conv1(self.act(self.norm1(x)))

        # add
        if self.temb_proj is not None:
            if self.cond_type == "ddpm":
                v = self.temb_proj(self.act(temb))[:, :, None]
                x_res = self.norm2(x_res + v)
            else:
                w, b = self.temb_proj(self.act(temb))[:, :, None].chunk(2, dim=1)
                x_res = self.norm2(x_res) * (w + 1) + b
        else:
            x_res = self.norm2(x_res)

        # forward conv2
        x_res = self.conv2(self.act(x_res))

        # shortcut
        x_skip = self.shortcut(x)

        # shortcut
        x = x_res + x_skip

        # additional mlp
        if self.mlp is not None:
            x_skip = self.mlp_skip(x)
            x = self.mlp(x) + x_skip

        return x

    def extra_repr(self) -> str:
        if self.temb_proj is not None:
            s = [f'(cond_type): {self.cond_type}']
        else:
            s = []
        return ', '.join(s)


class SpectralResidualBlock2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes_height,
                 modes_width,
                 temb_ch:int=None,
                 act=nn.GELU(),
                 use_mlp=False,
                 mlp:dict={'expansion': 1.0, 'dropout': 0},
                 norm=None,
                 skip='identity',
                 use_pointwise_op=False,
                 cond_type="improved",
                 **kwargs,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.act = act
        assert cond_type in ["ddpm", "improved"]
        self.cond_type = cond_type

        if temb_ch is not None and temb_ch > 0:
            self.temb_proj = nn.Linear(temb_ch, out_channels if cond_type == "ddpm" else out_channels*2)
        else:
            self.temb_proj = None
        self.norm1 = norm(in_channels, modes_height=modes_height) if norm is not None else nn.Identity()
        self.conv1 = SpectralConv2d(
                in_channels, out_channels, modes_height, modes_width,
                skip=use_pointwise_op,
                )
        self.norm2 = norm(out_channels, modes_height=modes_height) if norm is not None else nn.Identity()
        self.conv2 = SpectralConv2d(
                out_channels, out_channels, modes_height, modes_width,
                skip=use_pointwise_op,
                )

        if in_channels != out_channels:
            skip = 'linear'
        self.shortcut = skip_connection(in_channels, out_channels, type=skip, n_dim=2)

        if use_mlp:
            mlp = argparse.Namespace(**mlp)
            self.mlp = MLP(
                in_features=out_channels,
                hidden_features=int(round(out_channels*mlp.expansion)),
                n_dim=2,
                dropout=mlp.dropout,
            )
            self.mlp_skip = skip_connection(out_channels, out_channels, type=skip, n_dim=2)
        else:
            self.mlp = None

    def forward(self, x, temb=None):
        # forward conv1
        x_res = self.conv1(self.act(self.norm1(x)))

        # add timestep embedding
        if self.temb_proj is not None:
            if self.cond_type == "ddpm":
                v = self.temb_proj(self.act(temb))[:, :, None, None]
                x_res = self.norm2(x_res + v)
            else:
                w, b = self.temb_proj(self.act(temb))[:, :, None, None].chunk(2, dim=1)
                x_res = self.norm2(x_res) * (w + 1) + b
        else:
            x_res = self.norm2(x_res)

        # forward conv2
        x_res = self.conv2(self.act(x_res))

        # shortcut
        x_skip = self.shortcut(x)

        # shortcut
        x = x_res + x_skip

        # additional mlp
        if self.mlp is not None:
            x_skip = self.mlp_skip(x)
            x = self.mlp(x) + x_skip

        return x

    def extra_repr(self) -> str:
        if self.temb_proj is not None:
            s = [f'(cond_type): {self.cond_type}']
        else:
            s = []
        return ', '.join(s)

