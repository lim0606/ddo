import torch
import torch.nn as nn

from lib.models.unet import TimestepEmbedding
from lib.models.layers import PositionalEmbedding1d, PositionalEmbedding2d
from lib.models.resblocks import SpectralResidualBlock1d, SpectralResidualBlock2d
from lib.models.contconv import preprocessing, ContConv1d, ContConv2d
from lib.models.contconv_unet import Null


# class FNO1d(nn.Module):
#     def __init__(self,
#                  modes,
#                  ch,
#                  in_channels=1,
#                  num_layers=4,
#                  act=nn.GELU(), #F.gelu,
#                  #joint_factorization=False,
#                  rank=1.0, #0.1,
#                  factorization=None, #'tucker',
#                  fixed_rank_modes=False,
#                  fft_norm='forward',
#                  implementation='factorized',
#                  use_mlp=False,
#                  mlp:dict={'expansion': 1.0, 'dropout': 0},
#                  norm=None,
#                  skip='soft-gating', #'identity',
#                  pos_dim=1,
#                  ):
#         super().__init__()
# 
#         self.in_channels = in_channels
#         self.modes = modes
#         self.ch = ch
#         #self.pos_range = pos_range
#         self.act = act
#         self.pos_dim = pos_dim
# 
#         # Timestep embedding
#         self.temb_ch = ch * 4
#         self.temb_net = TimestepEmbedding(
#             embedding_dim=ch, #self.temb_ch,
#             hidden_dim=self.temb_ch,
#             output_dim=self.temb_ch,
#             pos_dim=1,
#             act=act,
#         )
#         self.pemb_net = PositionalEmbedding1d(
#             embedding_dim=ch, #self.temb_ch,
#             hidden_dim=self.temb_ch,
#             output_dim=self.temb_ch,
#             pos_dim=pos_dim,
#             act=act,
#         )
# 
#         convs = [FNOResidualBlock1d(
#                     in_channels=self.ch+self.temb_ch+1,
#                     out_channels=self.ch,
#                     modes_height=self.modes,
#                     temb_ch=self.temb_ch,
#                     act=act,
#                     #joint_factorization=joint_factorization,
#                     rank=rank,
#                     factorization=factorization,
#                     fixed_rank_modes=fixed_rank_modes,
#                     fft_norm=fft_norm,
#                     implementation=implementation,
#                     use_mlp=use_mlp,
#                     mlp=mlp,
#                     skip=skip,
#                     input_activation=False,
#             )]
#         convs += [FNOResidualBlock1d(
#                     in_channels=self.ch,
#                     out_channels=self.ch,
#                     modes_height=self.modes,
#                     temb_ch=self.temb_ch,
#                     act=act,
#                     #joint_factorization=joint_factorization,
#                     rank=rank,
#                     factorization=factorization,
#                     fixed_rank_modes=fixed_rank_modes,
#                     fft_norm=fft_norm,
#                     implementation=implementation,
#                     use_mlp=use_mlp,
#                     mlp=mlp,
#                     skip=skip,
#             ) for i in range(1, num_layers)]
#         self.convs = nn.ModuleList(convs)
# 
#         self.lifting = nn.Conv1d(in_channels, self.ch, kernel_size=1)
#         self.projection = nn.Sequential(
#             act,
#             nn.Conv1d(self.ch, in_channels, kernel_size=1),
#         )
# 
#     def _compute_module(self, module, x, temb):
#         for m in module:
#             x = m(x, temb=temb)
#         return x
# 
#     def forward(self, x, temp, v, **kwargs):
#         """
#         x: bsz x x_dim x height
#         temp: bsz
#         v: bsz x v_dim x height
#         """
#         # # init
#         assert x.shape[-1] == v.shape[-1]
# 
#         # Timestep embedding
#         temb = self.temb_net(temp)
#         pemb = self.pemb_net(v)
# 
#         # lift
#         x = torch.cat([self.lifting(x),
#                        self.act(pemb),
#                        self.act(temb)[:, :, None].expand(temb.shape[0], temb.shape[1], x.shape[-1]),
#                       ], dim=1)
# 
#         # forward
#         x = self._compute_module(self.convs, x, temb)
# 
#         # project
#         x = self.projection(x)
# 
#         return x


class FNOpp1d(nn.Module):
    def __init__(self,
                 modes_height,
                 ch,
                 in_channels=1,
                 num_layers=4,
                 act=nn.GELU(), #F.gelu,
                 # joint_factorization=False,
                 # rank=1.0, #0.1,
                 # factorization=None, #'tucker',
                 # fixed_rank_modes=False,
                 # fft_norm='forward',
                 # implementation='factorized',
                 use_mlp=False,
                 mlp:dict={'expansion': 1.0, 'dropout': 0},
                 norm=None,
                 skip='identity', #'soft-gating',
                 use_pointwise_op=False, #True,
                 use_time_embedding=True,
                 use_positional_embedding=True,
                 pos_dim=1,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.modes_height = modes_height
        self.ch = ch
        #self.pos_range = pos_range
        self.act = act
        self.pos_dim = pos_dim
        self.use_time_embedding = use_time_embedding
        self.use_positional_embedding = use_positional_embedding

        # Timestep embedding
        self.temb_ch = ch * 4 if self.use_time_embedding else 0
        if self.use_time_embedding:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch, #self.temb_ch,
                hidden_dim=self.temb_ch,
                output_dim=self.temb_ch,
                pos_dim=1,
                act=act,
            )
        self.pemb_ch = ch * 4 if self.use_positional_embedding else 0
        if self.use_positional_embedding:
            self.pemb_net = PositionalEmbedding1d(
                embedding_dim=ch, #self.temb_ch,
                hidden_dim=self.pemb_ch,
                output_dim=self.pemb_ch,
                pos_dim=pos_dim,
                act=act,
            )

        self.lifting = nn.Conv1d(in_channels, self.ch, kernel_size=1)

        convs = [SpectralResidualBlock1d(
                    in_channels=self.ch+self.pemb_ch,
                    out_channels=self.ch,
                    modes_height=self.modes_height,
                    temb_ch=self.temb_ch,
                    act=act,
                    #joint_factorization=joint_factorization,
                    #rank=rank,
                    #factorization=factorization,
                    #fixed_rank_modes=fixed_rank_modes,
                    #fft_norm=fft_norm,
                    #implementation=implementation,
                    use_mlp=use_mlp,
                    mlp=mlp,
                    skip=skip,
                    use_pointwise_op=use_pointwise_op,
            )]
        convs += [SpectralResidualBlock1d(
                    in_channels=self.ch,
                    out_channels=self.ch,
                    modes_height=self.modes_height,
                    temb_ch=self.temb_ch,
                    act=act,
                    #joint_factorization=joint_factorization,
                    #rank=rank,
                    #factorization=factorization,
                    #fixed_rank_modes=fixed_rank_modes,
                    #fft_norm=fft_norm,
                    #implementation=implementation,
                    use_mlp=use_mlp,
                    mlp=mlp,
                    skip=skip,
                    use_pointwise_op=use_pointwise_op,
            ) for i in range(1, num_layers)]
        self.convs = nn.ModuleList(convs)

        self.projection = nn.Sequential(
            act,
            nn.Conv1d(self.ch, in_channels, kernel_size=1),
        )

    def _compute_module(self, module, x, temb):
        for m in module:
            x = m(x, temb=temb)
        return x

    def forward(self, x, temp=None, v=None, **kwargs):
        """
        x: bsz x x_dim x height
        temp: bsz
        v: bsz x v_dim x height
        """
        # Timestep embedding
        if self.use_time_embedding:
            temb = self.temb_net(temp)
        else:
            temb = None

        # lift
        if self.use_positional_embedding:
            assert x.shape[-1] == v.shape[-1]
            pemb = self.pemb_net(v)
            x = torch.cat([self.lifting(x), self.act(pemb)], dim=1)
        else:
            x = self.lifting(x)

        # forward
        x = self._compute_module(self.convs, x, temb)

        # project
        x = self.projection(x)

        return x


class FNOpp2d(nn.Module):
    def __init__(self,
                 modes_height,
                 modes_width,
                 ch,
                 in_channels=1,
                 fc_channels=64,
                 num_layers=4,
                 act=nn.GELU(), #F.gelu,
                 # joint_factorization=False,
                 # rank=1.0, #0.1,
                 # factorization=None, #'tucker',
                 # fixed_rank_modes=False,
                 # fft_norm='forward',
                 # implementation='factorized',
                 use_mlp=False,
                 mlp:dict={'expansion': 1.0, 'dropout': 0},
                 norm=None,
                 skip='soft-gating', #'identity',
                 use_pointwise_op=True,
                 #pos_range=(0., 1.),
                 pos_dim=2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.ch = ch
        #self.pos_range = pos_range
        self.act = act
        self.pos_dim = pos_dim

        # Timestep embedding
        self.temb_ch = ch * 4
        self.temb_net = TimestepEmbedding(
            embedding_dim=ch,
            hidden_dim=self.temb_ch,
            output_dim=self.temb_ch,
            pos_dim=1,
            act=act,
        )
        self.pemb_net = PositionalEmbedding2d(
            embedding_dim=ch,
            hidden_dim=self.temb_ch,
            output_dim=self.temb_ch,
            pos_dim=pos_dim,
            act=act,
        )

        self.lifting = nn.Conv2d(in_channels, self.ch, kernel_size=1)

        convs = [SpectralResidualBlock2d(
                    in_channels=self.ch+self.temb_ch,
                    out_channels=self.ch,
                    modes_height=self.modes_height,
                    modes_width=self.modes_width,
                    pemb_ch=self.temb_ch,
                    temb_ch=self.temb_ch,
                    fc_channels=fc_channels,
                    act=act,
                    # joint_factorization=joint_factorization,
                    # rank=rank,
                    # factorization=factorization,
                    # fixed_rank_modes=fixed_rank_modes,
                    # fft_norm=fft_norm,
                    # implementation=implementation,
                    use_mlp=use_mlp,
                    mlp=mlp,
                    skip=skip,
                    use_pointwise_op=use_pointwise_op,
            )]
        convs += [SpectralResidualBlock2d(
                    in_channels=self.ch,
                    out_channels=self.ch,
                    modes_height=self.modes_height,
                    modes_width=self.modes_width,
                    pemb_ch=self.temb_ch,
                    temb_ch=self.temb_ch,
                    fc_channels=fc_channels,
                    act=act,
                    # joint_factorization=joint_factorization,
                    # rank=rank,
                    # factorization=factorization,
                    # fixed_rank_modes=fixed_rank_modes,
                    # fft_norm=fft_norm,
                    # implementation=implementation,
                    use_mlp=use_mlp,
                    mlp=mlp,
                    skip=skip,
                    use_pointwise_op=use_pointwise_op,
            ) for i in range(1, num_layers)]
        self.convs =  nn.ModuleList(convs)

        self.projection = nn.Sequential(
            act,
            nn.Conv2d(self.ch, in_channels, kernel_size=1),
        )

    def _compute_module(self, module, x, temb):
        for m in module:
            x = m(x, temb=temb)
        return x

    def forward(self, x, temp, v, **kwargs):
        """
        x: bsz x x_dim x height x width
        temp: bsz
        v: bsz x v_dim x height x width
        """
        # init
        assert x.shape[-1] == v.shape[-1] #and x.shape[-2] == v.shape[-2]

        # Timestep embedding
        temb = self.temb_net(temp)
        pemb = self.pemb_net(v)

        # lift
        x = torch.cat([self.lifting(x), self.act(pemb)], dim=1)

        # forward
        x = self._compute_module(self.convs, x, temb)

        # project
        x = self.projection(x)

        # return
        return x
