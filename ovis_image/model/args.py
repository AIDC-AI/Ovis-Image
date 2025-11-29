# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from dataclasses import dataclass, field

from ovis_image.model.autoencoder import AutoEncoderParams


@dataclass
class OvisImageModelArgs:
    in_channels: int = 64
    out_channels: int = 64
    context_in_dim: int = 512
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    double_block_type: str = "DoubleStreamBlock"
    depth_single_blocks: int = 38
    axes_dim: tuple = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    activation: str = "gelu_tanh"
    """activation: gelu_tanh or swiglu"""
    norm: str = "layernorm"
    """norm: layernorm or rmsnorm"""
    autoencoder_params: AutoEncoderParams = field(default_factory=AutoEncoderParams)
