import torch
import torch.nn as nn

from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F


def make_mlp(dim_list, activation_list, batch_norm=False, dropout=0):
    """
    Generates MLP network:

    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)

    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        activation = activation_list[index]
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))
        index += 1
    return nn.Sequential(*layers)


class Combiner(nn.Module):
    def __init__(
        self,
        command_dim: int = 768,
        features_channels: int = 256,
        combination_method="channel_attention",
        kernel_size=1,
        shared_command_fusion=True,
        num_scales=-1
    ):
        super(Combiner, self).__init__()
        self.combination_method = combination_method
        self.shared_command_fusion = shared_command_fusion

        if combination_method == "channel_attention":
            if shared_command_fusion:
                self.features_projector = nn.Conv2d(
                    in_channels=features_channels,
                    out_channels=features_channels,
                    kernel_size=1,
                )
                self.command_projector = make_mlp(
                    dim_list=[command_dim, 512, features_channels],
                    activation_list=["relu", None],
                )
            else:
                self.features_projector = nn.ModuleList([nn.Conv2d(
                    in_channels=features_channels,
                    out_channels=features_channels,
                    kernel_size=1,
                ) for _ in range(num_scales)])

                self.command_projector = [make_mlp(
                    dim_list=[command_dim, 512, features_channels],
                    activation_list=["relu", None],
                ) for _ in range(num_scales)]

        elif combination_method == "text2conv":
            if shared_command_fusion:
                self.command_projector = make_mlp(
                    dim_list=[command_dim, 512, features_channels ** 2 * kernel_size ** 2],
                    activation_list=["relu", None],
                )
            else:
                self.command_projector = nn.ModuleList([
                    make_mlp(
                        dim_list=[command_dim, 512, features_channels ** 2 * kernel_size ** 2],
                        activation_list=["relu", None],
                    )
                    for _ in range(num_scales)
                ])

            self.kernel_size = kernel_size
            self.feature_channels = features_channels

    def create_conv_filter(self, command, ix=None):
        B = command.shape[0]
        if ix is None:
            assert self.shared_command_fusion is True, "Did not pass an index whilest shared_command_fusion is set to false!"
            return self.command_projector(command).view(
                B,
                self.feature_channels,
                self.feature_channels,
                self.kernel_size,
                self.kernel_size,
            )
        else:
            return self.command_projector[ix](command).view(
                B,
                self.feature_channels,
                self.feature_channels,
                self.kernel_size,
                self.kernel_size,
            )

    def forward(self, features, command_embedding, ix=None):

        if self.combination_method == "channel_attention":
            batch, c, h, w = features.shape
            if self.shared_command_fusion:
                features = self.features_projector(features)
                command_embedding = self.command_projector(command_embedding)
            else:
                features = self.features_projector[ix](features)
                command_embedding = self.command_projector[ix](command_embedding)
            bmm_res = command_embedding.unsqueeze(1).bmm(features.view(batch, c, h * w))
            sftm = torch.softmax(bmm_res, dim=-1).view(batch, 1, h, w)
            features = sftm * features

        elif self.combination_method == "text2conv":
            B = command_embedding.shape[0]
            tmp = []
            for b in range(B):
                tmp.append(
                    F.conv2d(features[b].unsqueeze(0), command_embedding[b], padding=(self.kernel_size-1)//2)
                )
            features = torch.cat(tmp, 0)

        return features


class ChannelProjectionCombiner(nn.Module):
    def __init__(self, command_dim: int = 768, features_channels: int = 256):
        super(ChannelProjectionCombiner, self).__init__()
        self.features_projector = nn.Conv2d(
            in_channels=features_channels, out_channels=features_channels, kernel_size=1
        )
        self.command_projector = make_mlp(
            dim_list=[command_dim, 512, features_channels],
            activation_list=["relu", None],
        )

    def forward(self, features, command_embedding):
        batch, c, h, w = features.shape
        features = self.features_projector(features)
        command_embedding = self.command_projector(command_embedding)
        bmm_res = command_embedding.unsqueeze(1).bmm(features.view(batch, c, h * w))
        sftm = torch.softmax(bmm_res, dim=-1).view(batch, 1, h, w)
        features = sftm * features
        return features


class Attention(nn.Module):
    def __init__(self, command_dim, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(command_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Conv2d(dim, self.inner_dim, 2, 1, bias=False)

        self.attend = nn.Softmax(dim=-1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, patches, command):
        b, c, h, w, heads = *patches.shape, self.heads

        q = self.to_q(command)
        kv = self.to_kv(patches).chunk(2, dim=1)
        q2 = rearrange(q, "b (h d) -> b h d", h=heads)
        k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), kv)

        dots = einsum("b h d, b h j d -> b h j", q2, k) * self.scale
        softm = self.attend(dots)

        out = softm.unsqueeze(-1) * v
        out = rearrange(out, "b h (i j) d -> b i j (h d)", h=heads, i=h, j=w)
        return self.to_out(out).permute(0, 3, 1, 2)


class AttentionCombiner(nn.Module):
    def __init__(
        self,
        command_dim: int = 768,
        features_channels: int = 256,
    ):
        super(AttentionCombiner, self).__init__()

        self.att = Attention(command_dim=command_dim, dim=features_channels)
        self.norm = nn.InstanceNorm2d(num_features=features_channels)

    def forward(self, features, command_embedding):
        attn_output = self.att(features, command_embedding)
        features = self.norm(features + attn_output)
        return features


class MultiHeadAttentionCombiner(nn.Module):
    def __init__(
        self, command_dim: int = 768, features_channels: int = 256, n_heads: int = 4
    ):
        super(MultiHeadAttentionCombiner, self).__init__()

        self.features_projector = nn.Conv2d(
            in_channels=features_channels, out_channels=features_channels, kernel_size=1
        )
        self.command_projector = nn.Sequential(
            nn.Linear(command_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_channels),
        )
        self.att = nn.MultiheadAttention(
            embed_dim=features_channels, num_heads=n_heads, batch_first=True
        )

        self.att = Attention(command_dim=command_dim, dim=features_channels)
        self.norm = nn.InstanceNorm2d(num_features=features_channels)

    def forward(self, features, command_embedding):
        batch, c, h, w = features.shape
        features = self.features_projector(features)
        command_embedding = self.command_projector(command_embedding)
        features = features.view(batch, c, h * w).permute(0, 2, 1)
        command_embedding = command_embedding.unsqueeze(1)
        attn_output, attn_output_weights = self.att(
            features, command_embedding, command_embedding
        )

        attn_output = attn_output.permute(0, 2, 1).view(batch, c, h, w)
        features = features.permute(0, 2, 1).view(batch, c, h, w)
        # attn_output_weights = attn_output_weights.permute(0, 2, 1).view(batch, 1,  h, w)
        features = self.norm(features + attn_output)
        # features = attn_output
        # features = features * attn_output_weights
        return features
