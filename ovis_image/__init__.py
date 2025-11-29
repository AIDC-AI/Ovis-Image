# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
from ovis_image.model.args import OvisImageModelArgs
from ovis_image.model.autoencoder import AutoEncoderParams
from ovis_image.model.model import OvisImageModel

__all__ = [
    "OvisImageModelArgs",
    "OvisImageModel",
    "ovis_image_configs",
]


ovis_image_configs = {
    "ovis-image-7b": OvisImageModelArgs(
        in_channels=64,
        out_channels=64,
        context_in_dim=2048,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=6,
        double_block_type="DoubleStreamBlock",
        depth_single_blocks=27,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        activation = "swiglu",
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

