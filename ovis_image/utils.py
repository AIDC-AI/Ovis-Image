# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import torch
from torch import Tensor

def generate_txt_ids(encodings, time_id=0):
    txt_ids = torch.zeros(encodings.shape[0], encodings.shape[1], 3)
    txt_ids[..., 1] = txt_ids[..., 1] + torch.arange(encodings.shape[1])[None, :]
    txt_ids[..., 2] = txt_ids[..., 2] + torch.arange(encodings.shape[1])[None, :]
    txt_ids[..., 0] = time_id
    return txt_ids


def generate_noise_latent(
    bsz: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
    latent_channel = None,
) -> Tensor:
    """Generate noise latents for the flow model. The random seed will be set at the begining of training.

    Args:
        bsz (int): batch_size.
        height (int): The height of the image.
        width (int): The width of the image.
        device (str | torch.device): The device to use.
        dtype (torch.dtype): The dtype to use.

    Returns:
        Tensor: The noise latents.
            Shape: [num_samples, LATENT_CHANNELS, height // IMG_LATENT_SIZE_RATIO, width // IMG_LATENT_SIZE_RATIO]

    """
    LATENT_CHANNELS, IMAGE_LATENT_SIZE_RATIO = 16, 8
    if latent_channel is not None:
        LATENT_CHANNELS = latent_channel
    return torch.randn(
        bsz,
        LATENT_CHANNELS,
        height // IMAGE_LATENT_SIZE_RATIO,
        width // IMAGE_LATENT_SIZE_RATIO,
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)


def pack_latents(x: Tensor) -> Tensor:
    """
    Rearrange latents from an image-like format into a sequence of patches.
    Equivalent to `einops.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`.

    Args:
        x (Tensor): The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]

    Returns:
        Tensor: The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    b, c, latent_height, latent_width = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH

    # [b, c, h*ph, w*ph] -> [b, c, h, w, ph, pw]
    x = x.unfold(2, PATCH_HEIGHT, PATCH_HEIGHT).unfold(3, PATCH_WIDTH, PATCH_WIDTH)

    # [b, c, h, w, ph, PW] -> [b, h, w, c, ph, PW]
    x = x.permute(0, 2, 3, 1, 4, 5)

    # [b, h, w, c, ph, PW] -> [b, h*w, c*ph*PW]
    return x.reshape(b, h * w, c * PATCH_HEIGHT * PATCH_WIDTH)


def unpack_latents(x: Tensor, latent_height: int, latent_width: int) -> Tensor:
    """
    Rearrange latents from a sequence of patches into an image-like format.
    Equivalent to `einops.rearrange("b (h w) (c ph pw) -> b c (h ph) (w pw)")`.

    Args:
        x (Tensor): The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
        latent_height (int): The height of the unpacked latents.
        latent_width (int): The width of the unpacked latents.

    Returns:
        Tensor: The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    b, _, c_ph_pw = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH
    c = c_ph_pw // (PATCH_HEIGHT * PATCH_WIDTH)

    # [b, h*w, c*ph*pw] -> [b, h, w, c, ph, pw]
    x = x.reshape(b, h, w, c, PATCH_HEIGHT, PATCH_WIDTH)

    # [b, h, w, c, ph, pw] -> [b, c, h, ph, w, pw]
    x = x.permute(0, 3, 1, 4, 2, 5)

    # [b, c, h, ph, w, pw] -> [b, c, h*ph, w*pw]
    return x.reshape(b, c, h * PATCH_HEIGHT, w * PATCH_WIDTH)

