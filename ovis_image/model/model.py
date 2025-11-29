# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import torch
from torch import nn, Tensor

from ovis_image.model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from ovis_image.model.args import OvisImageModelArgs


class OvisImageModel(nn.Module):

    def __init__(self, model_args: OvisImageModelArgs):
        super().__init__()

        self.model_args = model_args

        self.in_channels = model_args.in_channels
        self.out_channels = model_args.out_channels
        if model_args.hidden_size % model_args.num_heads != 0:
            raise ValueError(
                f"Hidden size {model_args.hidden_size} must be divisible by num_heads {model_args.num_heads}"
            )
        pe_dim = model_args.hidden_size // model_args.num_heads
        if sum(model_args.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {model_args.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = model_args.hidden_size
        self.num_heads = model_args.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=model_args.theta, axes_dim=model_args.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.semantic_txt_norm = nn.RMSNorm(model_args.context_in_dim, eps=1e-6)
        self.semantic_txt_in = nn.Linear(model_args.context_in_dim, self.hidden_size, bias=True)

        if model_args.norm == "layernorm":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = nn.RMSNorm

        DoubleBlock = DoubleStreamBlock

        self.double_blocks = nn.ModuleList(
            [
                DoubleBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=model_args.mlp_ratio,
                    qkv_bias=model_args.qkv_bias,
                    activation=model_args.activation,
                    norm_layer=norm_layer,
                )
                for _ in range(model_args.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, 
                    self.num_heads, 
                    mlp_ratio=model_args.mlp_ratio,
                    qkv_bias=model_args.qkv_bias,
                    activation=model_args.activation,
                    norm_layer=norm_layer,
                )
                for _ in range(model_args.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size, 
            1, 
            self.out_channels, 
            norm_layer=norm_layer,
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        txt = self.semantic_txt_norm(txt)
        txt = self.semantic_txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


