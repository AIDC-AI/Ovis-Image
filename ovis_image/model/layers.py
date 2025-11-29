# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn, Tensor

from ovis_image.model.ops import attention, rope


class EmbedND(nn.Module):

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    @torch.no_grad()
    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        # bs x 1 x 512 x 64 x 2 x 2
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    with torch.device(t.device):
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def init_weights(self, init_std: float = 0.02):
        nn.init.normal_(self.in_layer.weight, std=init_std)
        nn.init.constant_(self.in_layer.bias, 0)
        nn.init.normal_(self.out_layer.weight, std=init_std)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def init_weights(self):
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def init_weights(self):
        for layer in (self.qkv, self.proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        self.norm.init_weights()

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


class YakMLP(nn.Module):
    # Use SwiGLU
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def init_weights(self):
        for layer in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def build_mlp(hidden_size, intermediate_size, activation = "gelu_tanh"):
    if activation == "gelu_tanh":
        mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size, bias=True),
        )
    else:
        mlp = YakMLP(hidden_size, intermediate_size)
    return mlp


def init_mlp(mlp, activation = "gelu_tanh"):
    if activation == "gelu_tanh":
        for layer in (mlp[0], mlp[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    else:
        mlp.init_weights()


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, multiples: int = 1):
        super().__init__()
        assert multiples in [1, 2, 3]
        self.multiples = multiples
        self.multiplier = 3 * multiples
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
        self.act = nn.SiLU()

    def init_weights(self):
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, vec: Tensor):
        out = self.lin(self.act(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )
        if self.multiples == 1:
            return ModulationOut(*out[:3])
        elif self.multiples == 2:
            return (
                ModulationOut(*out[:3]),
                ModulationOut(*out[3:]),
            )
        elif self.multiples == 3:
            return (
                ModulationOut(*out[:3]),
                ModulationOut(*out[3:6]),
                ModulationOut(*out[6:]),
            )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float, 
        qkv_bias: bool = False,
        activation: str = "gelu_tanh",
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.activation = activation
        self.img_mod = Modulation(hidden_size, multiples=2)
        self.img_norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = build_mlp(hidden_size, mlp_hidden_dim, activation)

        self.txt_mod = Modulation(hidden_size, multiples=2)
        self.txt_norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = build_mlp(hidden_size, mlp_hidden_dim, activation)

    def init_weights(self):
        # initialize all the nn.Linear submodules
        init_mlp(self.img_mlp, self.activation)
        init_mlp(self.txt_mlp, self.activation)

        # initialize Modulation layers, SelfAttention layers
        for layer in (self.img_attn, self.img_mod, self.txt_attn, self.txt_mod):
            layer.init_weights()

        # Reset parameters for Normalization layers
        for norm in (self.txt_norm1, self.txt_norm2, self.img_norm1, self.img_norm2):
            norm.reset_parameters()

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        activation: str = "gelu_tanh",
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.activation = activation

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if activation == "gelu_tanh":
            # qkv and mlp_in
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=qkv_bias)
        else:
            # qkv and mlp_in and mlp_gate
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=qkv_bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        
        if activation == "gelu_tanh":
            self.mlp_act = nn.GELU(approximate="tanh")
        else:
            self.mlp_act = nn.SiLU()
        self.modulation = Modulation(hidden_size, multiples=1)

    def init_weights(self):
        for layer in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        self.norm.init_weights()
        self.pre_norm.reset_parameters()
        self.modulation.init_weights()

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        if self.activation == "gelu_tanh":
            qkv, mlp = torch.split(
                self.linear1(x_mod), 
                [3 * self.hidden_size, self.mlp_hidden_dim], 
                dim=-1
            )
        else:
            qkv, mlp, mlp_gate = torch.split(
                self.linear1(x_mod), 
                [3 * self.hidden_size, self.mlp_hidden_dim, self.mlp_hidden_dim], 
                dim=-1
            )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        # compute attention
        attn = attention(q, k, v, pe=pe)

        if self.activation == "gelu_tanh":
            # compute activation in mlp stream, cat again and run second linear layer
            x = x + mod.gate * self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        else:
            x = x + mod.gate * self.linear2(
                torch.cat((attn, self.mlp_act(mlp_gate) * mlp), 2)
            )
        return x


class LastLayer(nn.Module):
    def __init__(
            self, 
            hidden_size: int, 
            patch_size: int, 
            out_channels: int, 
            norm_layer: nn.Module = nn.LayerNorm,
        ):
        super().__init__()
        self.norm_final = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def init_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        self.norm_final.reset_parameters()

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
