import functools
import torch
import torch.nn as nn

# from lib.models.unet import conv1d, conv2d
from lib.models.unet import TimestepEmbedding
# from lib.models.layers import normalize
from lib.models.layers import PositionalEmbedding1d, PositionalEmbedding2d
from lib.models.fourier import SpectralConv1d, SpectralConv2d, SpectralGroupNorm
from lib.models.resblocks import SpectralResidualBlock1d, SpectralResidualBlock2d
from lib.models.aliasfree_dct import Upsample1d, Upsample2d, Downsample1d, Downsample2d


def group_norm(out_ch, modes_height, num_groups=32):
    return SpectralGroupNorm(num_groups=num_groups, num_channels=out_ch, modes_height=modes_height, eps=1e-6, affine=True, cutoff=True)

def instance_norm(out_ch, modes_height):
    return group_norm(out_ch, modes_height, num_groups=out_ch)

def identity(*args, **kwargs):
    return nn.Identity()

def normalize(norm):
    if norm is None or norm == 'identity':
        return identity
    elif norm == 'instance_norm':
        return instance_norm
    elif norm == 'group_norm':
        return group_norm
    elif norm.startswith('group_norm'):
        num_groups = int(norm.split('_')[-1])
        return functools.partial(group_norm, num_groups=num_groups)
    else:
        raise ValueError


class SpectralDownsample1d(nn.Module):
    def __init__(
        self,
        in_ch,
        filter_size=9,
        #use_fft=False,
        with_conv:bool=False,
        modes_height=None,
        use_pointwise_op=False,
        **kwargs,
    ):
        super().__init__()
        self.down = Downsample1d(filter_size=filter_size)#, use_fft=use_fft)
        self.with_conv = with_conv
        if with_conv:
            self.conv = SpectralConv1d(
                in_ch, in_ch,
                modes_height=modes_height,
                skip=use_pointwise_op,
                )

    def forward(self, x):
        # batch_size, channels, height = x.shape
        out = self.down(x)
        if self.with_conv:
            out = self.conv(out)#, out_height=height//2)
        return out


class SpectralDownsample2d(nn.Module):
    def __init__(
        self,
        in_ch,
        filter_size=9,
        #use_fft=False,
        use_radial=False,
        with_conv:bool=False,
        modes_height=None,
        modes_width=None,
        use_pointwise_op=False,
    ):
        super().__init__()
        self.down = Downsample2d(filter_size=filter_size, use_radial=use_radial)#, use_fft=use_fft)
        self.with_conv = with_conv
        if with_conv:
            self.conv = SpectralConv2d(
                in_ch, in_ch,
                modes_height=modes_height,
                modes_width=modes_width,
                skip=use_pointwise_op,
                )

    def forward(self, x):
        # batch_size, channels, height, width = x.shape
        out = self.down(x)
        if self.with_conv:
            out = self.conv(out)#, out_height=height//2, out_width=width//2)
        return out


class SpectralUpsample1d(nn.Module):
    def __init__(
        self,
        in_ch,
        filter_size=9,
        #use_fft=False,
        with_conv:bool=False,
        modes_height=None,
        use_pointwise_op=False,
        **kwargs,
    ):
        super().__init__()
        self.up = Upsample1d(filter_size=filter_size)#, use_fft=use_fft)
        self.with_conv = with_conv
        if with_conv:
            self.conv = SpectralConv1d(
                in_ch, in_ch,
                modes_height=modes_height,
                skip=use_pointwise_op,
         )

    def forward(self, x):
        # batch_size, channels, height = x.shape
        out = self.up(x)
        if self.with_conv:
            out = self.conv(out)#, out_height=height*2)
        return out


class SpectralUpsample2d(nn.Module):
    def __init__(
        self,
        in_ch,
        filter_size=9,
        #use_fft=False,
        use_radial=False,
        with_conv:bool=False,
        modes_height=None,
        modes_width=None,
        use_pointwise_op=False,
    ):
        super().__init__()
        self.up = Upsample2d(filter_size=filter_size, use_radial=use_radial)#, use_fft=use_fft)
        self.with_conv = with_conv
        if with_conv:
            self.conv = SpectralConv2d(
                in_ch, in_ch,
                modes_height=modes_height,
                modes_width=modes_width,
                skip=use_pointwise_op,
                )

    def forward(self, x):
        # batch_size, channels, height, width = x.shape
        out = self.up(x)
        if self.with_conv:
            out = self.conv(out)#, out_height=height*2, out_width=width*2)
        return out


class FNOUNet(nn.Module):
    def __init__(self,
                 modes_height,
                 modes_width,
                 in_channels=1,
                 out_channels=None,
                 ch=64,
                 ch_mult=(1, 2, 4, 8), # channel multiplier
                 num_res_blocks=2,
                 attn_resolutions=(),
                 dropout=0.,
                 act=nn.SiLU(), #nn.GELU(),
                 norm=None, #group_norm,
                 skip='identity', #'soft-gating',
                 use_mlp=False,
                 mlp={'expansion': 1.0, 'dropout': 0},
                 resamp_with_conv=True,
                 use_pointwise_op=False,
                 use_time_embedding=True,
                 use_pos=False,
                 # use_positional_embedding=False,
                 base_resolution=None,
                 pos_dim=2,
                 **kwargs,
                 ):
        super().__init__()
        assert modes_height == modes_width
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None \
            else out_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.act = act
        self.norm = norm = normalize(norm)
        self.pos_dim = pos_dim
        self.use_time_embedding = use_time_embedding
        # self.use_positional_embedding = use_positional_embedding
        self.use_pos = use_pos

        if pos_dim == 1:
            conv_init = nn.Conv1d #conv1d
            PositionalEmbedding = PositionalEmbedding1d
            SpectralResidualBlock = SpectralResidualBlock1d
            # SelfAttention = SelfAttention1d
            SpectralDownsample = SpectralDownsample1d
            SpectralUpsample = SpectralUpsample1d
        elif pos_dim == 2:
            conv_init = nn.Conv2d #conv2d
            PositionalEmbedding = PositionalEmbedding2d
            SpectralResidualBlock = SpectralResidualBlock2d
            # SelfAttention = SelfAttention2d
            SpectralDownsample = SpectralDownsample2d
            SpectralUpsample = SpectralUpsample2d
        else:
            raise NotImplementedError

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        temb_ch = ch * 4 if self.use_time_embedding else 0
        # pemb_ch = ch * 4 if self.use_positional_embedding else 0
        pos_ch = pos_dim if self.use_pos else 0

        # Timestep embedding
        if self.use_time_embedding:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
                pos_dim=1,
                act=act,
            )
        # if self.use_positional_embedding:
        #     self.pemb_net = PositionalEmbedding(
        #         embedding_dim=ch,
        #         hidden_dim=pemb_ch,
        #         output_dim=pemb_ch,
        #         pos_dim=pos_dim,
        #         act=act,
        #     )

        # Begin
        self.lifting = conv_init(in_channels+pos_ch, ch, kernel_size=1, padding=0)

        # Downsampling
        unet_chs = [ch] #unet_chs = [ch + pemb_ch]
        in_ht = base_resolution if base_resolution is not None else modes_height
        assert in_ht % 2 ** (num_resolutions - 1) == 0, "in_ht doesn't satisfy the condition"
        in_ch = unet_chs[-1] #in_ch = ch #in_ch = ch + pemb_ch
        down_modules = []
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    SpectralResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        modes_height=min(in_ht, modes_height),
                        modes_width=min(in_ht, modes_height),
                        #pemb_ch=pemb_ch,
                        temb_ch=temb_ch,
                        act=act,
                        use_mlp=use_mlp,
                        mlp=mlp,
                        norm=norm,
                        skip=skip,
                        use_pointwise_op=use_pointwise_op,
                    )
                if in_ht in attn_resolutions:
                    raise NotImplementedError
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                in_ht //= 2
                block_modules['{}b_downsample'.format(i_level)] = \
                    SpectralDownsample(
                        out_ch,
                        modes_height=min(in_ht, modes_height),
                        modes_width=min(in_ht, modes_height),
                        with_conv=resamp_with_conv,
                        use_pointwise_op=use_pointwise_op,
                        # filter_size=min(in_ht, 8), #min(in_ht//2, 8),
                        # use_fft=True, #False if in_ht >= 8 else True,
                    )
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [
            SpectralResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                modes_height=min(in_ht, modes_height),
                modes_width=min(in_ht, modes_height),
                #pemb_ch=pemb_ch,
                temb_ch=temb_ch,
                act=act,
                use_mlp=use_mlp,
                mlp=mlp,
                norm=norm,
                skip=skip,
                use_pointwise_op=use_pointwise_op,
            )]
        if len(attn_resolutions) > 0:
            raise NotImplementedError
            mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [
            SpectralResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                modes_height=min(in_ht, modes_height),
                modes_width=min(in_ht, modes_height),
                #pemb_ch=pemb_ch,
                temb_ch=temb_ch,
                act=act,
                use_mlp=use_mlp,
                mlp=mlp,
                norm=norm,
                skip=skip,
                use_pointwise_op=use_pointwise_op,
            )]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                skip_ch = unet_chs.pop()
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    SpectralResidualBlock(
                        in_channels=in_ch + skip_ch,
                        out_channels=out_ch,
                        modes_height=min(in_ht, modes_height),
                        modes_width=min(in_ht, modes_height),
                        #pemb_ch=pemb_ch,
                        temb_ch=temb_ch,
                        act=act,
                        use_mlp=use_mlp,
                        mlp=mlp,
                        norm=norm,
                        skip=skip,
                        use_pointwise_op=use_pointwise_op,
                    )
                if in_ht in attn_resolutions:
                    raise NotImplementedError
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                in_ch = out_ch
            # Upsample
            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = \
                    SpectralUpsample(
                        out_ch,
                        modes_height=min(in_ht, modes_height),
                        modes_width=min(in_ht, modes_height),
                        with_conv=resamp_with_conv,
                        use_pointwise_op=use_pointwise_op,
                        # filter_size=min(in_ht, 8), #min(in_ht//2, 8),
                        # use_fft=True, #False if in_ht >= 8 else True,
                    )
                in_ht *= 2
            # convert list of modules to a module list, and append to a list
            up_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # End
        self.projection = nn.Sequential(
            # norm(in_ch, modes_height=min(in_ht, modes_height)) if norm is not None else nn.Identity(),
            act,
            conv_init(in_ch, out_channels, kernel_size=1, padding=0), #, init_scale=0.),
        )

    def _compute_module(self, module, x, temb):
        for m in module:
            x = m(x, temb)
        return x

    def forward(self, x, temp=None, v=None, **kwargs):
        """
        x: bsz x x_dim x height x width
        temp: bsz
        v: bsz x v_dim x height x width
        """
        # Timestep embedding
        if self.use_time_embedding:
            temb = self.temb_net(temp)
        else:
            temb = None

        # Begin
        # if self.use_positional_embedding:
        #     assert x.shape[-1] == v.shape[-1]# and x.shape[-2] == v.shape[-2]
        #     pemb = self.pemb_net(v)
        #     h = torch.cat([self.lifting(x), self.act(pemb)], dim=1)
        if self.use_pos:
            h = self.lifting(torch.cat([x, v], dim=1))
        else:
            h = self.lifting(x)

        # Downsampling
        in_ht = h.shape[-1]
        hs = [h]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(hs[-1], temb)
                if h.size(-1) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                    h = attn_block(h, temb)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                in_ht //= 2
                downsample = block_modules['{}b_downsample'.format(i_level)]
                hs.append(downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_module(self.mid_modules, h, temb)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            # Residual blocks for this resolution
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                if h.size(-1) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                    h = attn_block(h, temb)
            # Upsample
            if i_level != 0:
                in_ht *= 2
                upsample = block_modules['{}b_upsample'.format(i_level)]
                h = upsample(h)
        assert not hs

        # End
        h = self.projection(h)
        #assert list(h.size()) == [x.size(0), self.out_channels, x.size(2), x.size(3)]
        return h

class FNOUNet1d(FNOUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pos_dim=1, **kwargs)

class FNOUNet2d(FNOUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pos_dim=2, **kwargs)
