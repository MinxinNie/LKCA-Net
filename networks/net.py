import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einops
from timm.layers import DropPath

from networks.blocks_2d import PaCaBlock, MultiDWConv, deformableLKABlock, UpConv, ChannelAttentionCBAM
from backbone.network import MaxViT4Out_Small
from networks.blocks_2d import LayerNormProxy
from networks.deform_ops import DeformConv2d


class CNN(nn.Module):
    def __init__(
            self,
            embed_dims=[96, 192, 384, 768],
            clusters=[10, 10, 10, 10],
            kernel0=[5, 5, 5, 5],
            kernel1=[7, 7, 7, 7],
            dilation=[3, 3, 3, 3]
    ):
        super().__init__()
        self.num_stages = len(embed_dims)
        self.num_clusters = clusters
        for i in range(self.num_stages):
            channel_attention = ChannelAttentionCBAM(embed_dims[i])
            setattr(self, f"ca{i + 1}", channel_attention)
            mdw = MultiDWConv(embed_dims[i], embed_dims[i])
            setattr(self, f"mdw{i + 1}", mdw)
            dfm1 = deformableLKABlock(embed_dims[i], kernel0[i], kernel1[i], dilation[i])
            setattr(self, f"dfm1{i + 1}", dfm1)
            to_clusters = nn.Conv2d(embed_dims[i], clusters[i], 1, 1, 0) if clusters[i] > 0 else nn.Identity()
            setattr(self, f"to_clusters{i + 1}", to_clusters)
            proj = nn.Conv2d(embed_dims[i], embed_dims[i], 1)
            setattr(self, f"proj{i + 1}", proj)

    def forward(self, x_down):
        clusters = []
        conv_f = []
        for i in range(self.num_stages):
            x_shortcut = x_down[i].clone()
            mdw = getattr(self, f"mdw{i + 1}")
            x = mdw(x_down[i])
            dfm1 = getattr(self, f"dfm1{i + 1}")
            x = dfm1(x)

            proj = getattr(self, f"proj{i + 1}")
            x = proj(x)
            channel_attention = getattr(self, f"ca{i + 1}")
            x = x * channel_attention(x_shortcut)
            x = x + x_shortcut
            to_cluster = getattr(self, f"to_clusters{i + 1}")
            clusters.append(to_cluster(x))
            conv_f.append(x)

        return conv_f, clusters


class DecoderBlock(nn.Module):
    def __init__(
            self,
            depths,
            cluster,
            embed_dim,
            heads,
            expansion,
            drop_path_rate=0.0,
            attn_drop=0.0,
            drop=0.0,
            with_pos_embed=False,
            layer_scale=None,
            img_size=224,
    ):
        super().__init__()
        self.cluster = cluster
        self.blocks = nn.ModuleList()
        self.dwconv1 = nn.Conv2d(embed_dim, embed_dim, 1, bias=True, groups=embed_dim)
        self.to_clusters = nn.Conv2d(embed_dim, cluster, 1, 1, 0) if self.cluster > 0 else nn.Identity()
        self.self_attn = nn.ModuleList()
        for i in range(depths):
            block = PaCaBlock(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=expansion,
                drop_path=drop_path_rate,
                attn_drop=attn_drop,
                drop=drop,
                clusters=self.cluster,
                layer_scale=layer_scale,
                input_resolution=(
                    img_size, img_size
                ),
                with_pos_embed=True if i == 0 else False,
            )
            self.self_attn.append(block)

        self.cross_attn = nn.ModuleList()
        for i in range(depths):
            block = PaCaBlock(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=expansion,
                drop_path=drop_path_rate,
                attn_drop=attn_drop,
                drop=drop,
                clusters=self.cluster,
                layer_scale=layer_scale,
                input_resolution=(
                    img_size, img_size
                ),
                with_pos_embed=True if i == 0 else False,
            )
            self.cross_attn.append(block)

        self.proj = MultiDWConv(embed_dim, embed_dim)

        self.dwconv2 = nn.Conv2d(embed_dim, embed_dim, 1, bias=True, groups=embed_dim)

    def forward(self, x, z, f):

        B, C, H, W = x.shape
        x = self.dwconv1(x)  # up sample
        if self.cluster > 0:
            cluster = self.to_clusters(x)
            cluster = rearrange(cluster, "B M H W -> B M (H W)").softmax(dim=-1)
        else:
            cluster = rearrange(x, "B C H W -> B C (H W)").softmax(dim=-1)

        # SA
        x1 = x
        for block in self.self_attn:
            x, cluster = block(x, H, W, cluster)
        x += x1

        x2 = x
        for block in self.cross_attn:
            x, z = block(x, H, W, z)
        x += x2

        x += f
        x = self.proj(x) + x
        x = self.dwconv2(x)

        return x


class NET(nn.Module):
    def __init__(self, img_size=224, num_classes=9, clusters=[10, 10, 10, 10],
                 dims=[96, 192, 384, 768], depths=[2, 2, 4, 2], expansion=[4, 4, 4, 4],
                 heads=[3, 6, 12, 24], kernel0=[5, 5, 5, 5], kernel1=[7, 7, 7, 7], dilation=[3, 3, 3, 3], deep_supervision=True):
        super().__init__()
        self.img_size = img_size
        self.expansion = expansion
        self.num_classes = num_classes
        self.clusters = clusters
        self.deep_supervision = deep_supervision
        self.stages = len(dims)
        self.encoder = MaxViT4Out_Small(n_class=num_classes, img_size=img_size)

        self.bottleneck = nn.Sequential(*[
            nn.Conv2d(dims[-1], dims[-1] * 2, 1, 1, 0),
            nn.Conv2d(dims[-1] * 2, dims[-1], 1, 1, 0),
        ])

        self.conv_attn = CNN(dims, clusters, kernel0, kernel1, dilation)

        self.up_projs = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.stages):
            if i < self.stages - 1:
                self.up_projs.append(UpConv(dims[i + 1], dims[i]))

            self.concat_back_dim.append(nn.Conv2d(dims[i] * 2, dims[i], 1, 1, 0))
            decoder = DecoderBlock(depths=depths[i], cluster=clusters[i], embed_dim=dims[i], heads=heads[i],
                                   expansion=expansion[i], with_pos_embed=False, img_size=img_size // 2 ** (2 + i))
            setattr(self, f"decoder{i + 1}", decoder)
            output = nn.Conv2d(in_channels=dims[i], out_channels=self.num_classes, kernel_size=1, bias=False)
            setattr(self, f"output{i + 1}", output)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_encoder(self, x):

        f4, f3, f2, f1 = self.encoder(x)
        x_downsample = [f1, f2, f3, f4]
        x = self.bottleneck(f4)

        f, f_clusters = self.conv_attn(x_downsample)
        return x, x_downsample, f, f_clusters

    def forward_decoder(self, x, f, f_conv, f_cluster):
        outputs = []
        for i in reversed(range(self.stages)):
            # print(x.shape)
            if i != self.stages - 1:
                x = self.up_projs[i](x)
            x = torch.cat([x, f[i]], 1)
            x = self.concat_back_dim[i](x)
            decoder = getattr(self, f"decoder{i + 1}")
            x = decoder(x, f_cluster[i], f_conv[i])
            output = getattr(self, f"output{i + 1}")
            seg_out = output(x)
            if self.deep_supervision:
                outputs.append(F.interpolate(seg_out, scale_factor=int(4 * 2**i), mode='bilinear'))
        return outputs

    def forward(self, x):
        x, x_downsample, conv_f, z = self.forward_encoder(x)
        outputs = self.forward_decoder(x, x_downsample, conv_f, z)
        return outputs


