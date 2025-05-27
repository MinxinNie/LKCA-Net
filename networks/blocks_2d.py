import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.layers import DropPath
from timm.models.layers import to_2tuple, trunc_normal_
from torch import einsum
from torch.nn import LayerNorm

from networks.deform_ops import DeformConv2d


def c_rearrange(x, H, W, dim=1):
    channels_last = x.is_contiguous(memory_format=torch.channels_last)
    if dim == 1:
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
    elif dim == 2:
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
    elif dim == 3:
        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W)
    else:
        raise NotImplementedError

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        x = x.contiguous()
    return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttnDWConv(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=True, with_shortcut=False):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, 1, (kernel_size - 1) // 2, bias=bias, groups=dim
        )
        self.with_shortcut = with_shortcut

    def forward(self, x, H, W):
        shortcut = x
        x = c_rearrange(x, H, W)
        x = self.dwconv(x)
        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W)
        if self.with_shortcut:
            return x + shortcut
        return x


class AttnMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            drop=0.0,
            with_shortcut=False,

    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = AttnDWConv(hidden_features, with_shortcut=with_shortcut)

        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ChannelAttentionCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# class MultiDWConv(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels):
#         super().__init__()
#         self.in_c = in_channels // 3
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(self.in_c, self.in_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.in_c),
#             nn.GELU()
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(self.in_c, self.in_c, kernel_size=3, padding="same", dilation=2),
#             nn.BatchNorm2d(self.in_c),
#             nn.GELU()
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(self.in_c, self.in_c, kernel_size=3, padding="same", dilation=3),
#             nn.BatchNorm2d(self.in_c),
#             nn.GELU()
#         )
#
#         # 带有通道注意力的融合层
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(self.in_c * 3, self.in_c * 3, kernel_size=1),
#             nn.GELU(),
#             nn.BatchNorm2d(self.in_c * 3)
#         )
#
#     def forward(self, x):
#
#         in_c = self.in_c
#         x1 = x[:, :in_c, :, :]
#         x2 = x[:, in_c:in_c*2, :, :]
#         x3 = x[:, in_c*2:in_c*3, :, :]
#
#         x1 = self.conv1(x1)
#
#         x1 = F.dropout(x1, 0.1)
#         x2 = self.conv2(x2)
#
#         x2 = F.dropout(x2, 0.1)
#         x3 = self.conv3(x3)
#
#         x3 = F.dropout(x3, 0.1)
#         added = torch.concat((x1, x2),dim=1)
#         added = torch.concat((added, x3),dim=1)
#         x_out = self.conv4(added)
#
#         x_out = F.dropout(x_out, 0.1)
#         return x_out

class MultiDWConv(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1=  nn.Conv2d(in_channels, out_channels, 3, 1, padding="same",groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2,groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3,groups=in_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3=self.conv3(x)
        x3=F.gelu(x3)
        x3=F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out

class deformable_LKA(nn.Module):
    def __init__(self, dim, kernel0, kernel1, dilation):
        super().__init__()
        self.conv0 = DeformConv2d(dim, dim, kernel_size=kernel0, stride=1, padding=kernel0 // 2, groups=dim)
        self.conv_spatial = DeformConv2d(dim, dim, kernel_size=kernel1, stride=1, padding=(kernel1 // 2) * dilation,
                                         groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model, kernel0, kernel1, dilation):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model, kernel0, kernel1, dilation)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class deformableLKABlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel0,
                 kernel1,
                 dilation,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 ):
        super().__init__()
        self.norm1 = LayerNormProxy(dim)
        self.attn = deformable_LKA_Attention(dim, kernel0, kernel1, dilation)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNormProxy(dim)
        expansion = int(mlp_ratio)
        self.mlp = ConvMlp(dim, expansion, drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        y = self.norm2(y)
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class P2CConv2d(nn.Module):
    def __init__(
            self,
            dim,
            num_clusters,
            kernel_size=7,
            **kwargs,
    ) -> None:
        super().__init__()

        self.clustering = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=dim,
            ),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, num_clusters, 1, 1, 0, bias=False),
            Rearrange("B M H W -> B M (H W)"),
        )

    def forward(self, x, H, W):
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        return self.clustering(x)


class P2CMlp(nn.Module):
    def __init__(
            self, dim, num_clusters, mlp_ratio=4.0, **kwargs
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.clustering = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_clusters),  # TODO: train w/ bias=False
            Rearrange("B N M -> B M N"),
        )

    def forward(self, x, H, W):
        return self.clustering(x)


class PaCaLayer(nn.Module):
    """Patch-to-Cluster Attention Layer"""

    def __init__(
            self,
            dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=4.0,
            clusters=10,
            k_size=7,
            onsite_clustering=True,
            cluster_pos_embed=True,
            **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.num_clusters = clusters
        self.onsite_clustering = onsite_clustering
        self.cluster_pos_embed = cluster_pos_embed
        if self.num_clusters > 0:
            self.cluster_norm = nn.LayerNorm(dim)
            if self.onsite_clustering:
                self.clustering = P2CConv2d(
                    dim=dim,
                    num_clusters=self.num_clusters,
                    mlp_ratio=mlp_ratio,
                    kernel_size=k_size,
                )

            if self.cluster_pos_embed:
                self.cluster_pos_enc = nn.Parameter(
                    torch.zeros(1, self.num_clusters, dim)
                )
                trunc_normal_(self.cluster_pos_enc, std=0.02)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity()  # get attn weights for viz.

    def forward(self, x, H, W, z):
        # x: B N C

        if self.num_clusters > 0:
            if self.onsite_clustering:
                z_raw = self.clustering(x, H, W)  # B M N
                z = z_raw.softmax(dim=-1)
                # TODO: how to auto-select the 'meaningful' subset of clusters
            # c = z @ x  # B M C
            c = einsum("bmn,bnc->bmc", z, x)
            if self.cluster_pos_embed:
                c = c + self.cluster_pos_enc.expand(c.shape[0], -1, -1)
            c = self.cluster_norm(c)
        else:
            c = x

        x = rearrange(x, "B N C -> N B C")
        c = rearrange(c, "B M C -> M B C")

        x, attn = F.multi_head_attention_forward(
            query=x,
            key=c,
            value=c,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q.weight,
            k_proj_weight=self.k.weight,
            v_proj_weight=self.v.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_drop,
            out_proj_weight=self.proj.weight,
            out_proj_bias=self.proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=not self.training,  # for visualization
            average_attn_weights=False,
        )

        x = rearrange(x, "N B C -> B N C")

        if not self.training:
            attn = self.attn_viz(attn)

        x = self.proj_drop(x)
        return x, z


class PaCaBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop_path=0.0,
            attn_drop=0.0,
            drop=0.0,
            clusters=10,
            layer_scale=None,
            input_resolution=None,
            with_pos_embed=False,
            post_norm=False,
            onsite_clustering=True,
            cluster_pos_embed=True,
            **kwargs,
    ):
        super().__init__()

        self.post_norm = post_norm
        self.with_pos_embed = with_pos_embed
        self.input_resolution = input_resolution
        if self.with_pos_embed:
            assert self.input_resolution is not None
            self.input_resolution = to_2tuple(self.input_resolution)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.input_resolution[0] * self.input_resolution[1], dim)
            )
            self.pos_drop = nn.Dropout(p=drop)
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm1_before = nn.LayerNorm(dim)
        self.attn = PaCaLayer(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            mlp_ratio=mlp_ratio,
            clusters=clusters,
            onsite_clustering=onsite_clustering,
            cluster_pos_embed=cluster_pos_embed,
        )
        self.norm1_after = nn.LayerNorm(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2_before = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = AttnMlp(dim, hidden_dim, drop=drop)
        self.norm2_after = nn.LayerNorm(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x, H, W, z):
        # x: B N C
        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W).contiguous()
        if self.with_pos_embed:
            if self.input_resolution != (H, W):
                pos_embed = rearrange(self.pos_embed, "B (H W) C -> B C H W")
                pos_embed = F.interpolate(
                    pos_embed, size=(H, W), mode="bilinear", align_corners=True
                )
                pos_embed = rearrange(pos_embed, "B C H W -> B (H W) C")
            else:
                pos_embed = self.pos_embed

            x = self.pos_drop(x + pos_embed)

        a, z = self.attn(self.norm1_before(x), H, W, z)
        a = self.norm1_after(a)
        if not self.layer_scale:
            x = x + self.drop_path1(a)
            x = x + self.drop_path2(
                self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )
        else:
            x = x + self.drop_path1(self.gamma1 * a)
            x = x + self.drop_path2(
                self.gamma2 * self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )
        x = rearrange(x, "B (H W) C ->B C H W", H=H, W=W).contiguous()
        return x, z
