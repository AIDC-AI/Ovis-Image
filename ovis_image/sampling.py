# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
import math
import os
from typing import Callable, Optional
from PIL import ExifTags, Image
import torch
from torch import Tensor
from einops import rearrange, repeat

from ovis_image.dataset.image_util import build_img_ids
from ovis_image.model.autoencoder import AutoEncoder
from ovis_image.model.hf_embedder import OvisEmbedder
from ovis_image.model.model import OvisImageModel
from ovis_image.utils import (
    generate_noise_latent,
    pack_latents,
    unpack_latents,
    generate_txt_ids,
)



def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def sample_timesteps(batch_size, image_seq_len=None, base_shift=None, max_shift=None):
    if image_seq_len is None or base_shift is None or max_shift is None:
        logit_mean = 0
    else:
        logit_mean = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    logit_std = 1.0
    timesteps = torch.normal(
        mean=logit_mean, std=logit_std, size=(batch_size,)
    )
    timesteps = torch.nn.functional.sigmoid(timesteps)
    return timesteps


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def generate_image(
    device: torch.device,
    dtype: torch.dtype,
    model: OvisImageModel,
    prompt: str,
    autoencoder: AutoEncoder,
    ovis_tokenizer,
    ovis_encoder: OvisEmbedder,
    img_height: int = 256,
    img_width: int = 256,
    denoising_steps: int = 50,
    cfg_scale: float = 5.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sampling and save a single images from noise using a given prompt.
    For randomized noise generation, the random seend should already be set at the begining of training.
    Since we will always use the local random seed on this rank, we don't need to pass in the seed again.
    """

    # allow for packing and conversion to latent space. Use the same resolution as training time.
    img_height = 16 * (img_height // 16)
    img_width = 16 * (img_width // 16)

    enable_classifier_free_guidance = True

    # Tokenize the prompt. Unsqueeze to add a batch dimension.
    ovis_token_ids, ovis_token_mask = ovis_tokenizer.encode(prompt)
    ovis_encodings = ovis_encoder(
        ovis_token_ids.to(device=device), ovis_token_mask.to(device=device)
    )

    if enable_classifier_free_guidance:
        empty_ovis_token_ids, empty_ovis_token_mask = ovis_tokenizer.encode("")
        empty_ovis_encodings = ovis_encoder(
            empty_ovis_token_ids.to(device=device), empty_ovis_token_mask.to(device=device)
        )

    latents = generate_noise_latent(
        ovis_token_ids.shape[0],
        img_height, img_width, device, dtype, seed=seed,
        latent_channel=autoencoder.params.z_channels)

    img = denoise(
        device=device,
        dtype=dtype,
        model=model,
        latents=latents,
        denoising_steps=denoising_steps,
        ovis_encodings=ovis_encodings,
        enable_classifier_free_guidance=enable_classifier_free_guidance,
        empty_ovis_encodings=(
            empty_ovis_encodings if enable_classifier_free_guidance else None
        ),
        classifier_free_guidance_scale=cfg_scale,
    )

    img = autoencoder.decode(img)
    return img


def denoise(
    device: torch.device,
    dtype: torch.dtype,
    model: OvisImageModel,
    latents: torch.Tensor,
    denoising_steps: int,
    ovis_encodings: torch.Tensor,
    enable_classifier_free_guidance: bool = False,
    empty_ovis_encodings: torch.Tensor | None = None,
    classifier_free_guidance_scale: float | None = None,
) -> torch.Tensor:
    """
    Sampling images from noise using a given prompt, by running inference with trained model.
    Save the generated images to the given output path.
    """
    bsz = ovis_encodings.shape[0]
    _, latent_channels, latent_height, latent_width = latents.shape

    # create denoising schedule
    timesteps = get_schedule(denoising_steps, latent_height * latent_width, shift=True)

    # create positional encodings

    latent_pos_enc = build_img_ids(
        latent_height // 2, latent_width // 2,
    ).to(latents)
    latent_pos_enc = repeat(latent_pos_enc, 'l c -> bsz l c', bsz=bsz)
    ovis_txt_ids = generate_txt_ids(ovis_encodings, time_id=0).to(latents)

    if enable_classifier_free_guidance:
        ovis_encodings = torch.cat([empty_ovis_encodings, ovis_encodings], dim=0)
        latent_pos_enc = torch.cat([latent_pos_enc, latent_pos_enc], dim=0)
        ovis_txt_ids = torch.cat([ovis_txt_ids, ovis_txt_ids], dim=0)

    # convert img-like latents into sequences of patches
    latents = pack_latents(latents)

    # this is ignored for schnell
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        if enable_classifier_free_guidance:
            img = torch.cat([latents, latents], dim=0)
            t_vec = torch.full((bsz * 2,), t_curr, dtype=dtype, device=device)
        else:
            img = latents
            t_vec = torch.full((bsz,), t_curr, dtype=dtype, device=device)
        model_pred = model(
            img=img,
            img_ids=latent_pos_enc,
            txt=ovis_encodings,
            txt_ids=ovis_txt_ids,
            timesteps=t_vec,
        )
        if enable_classifier_free_guidance:
            pred_u, pred_c = model_pred.chunk(2)
            pred = pred_u + classifier_free_guidance_scale * (pred_c - pred_u)
        else:
            pred = model_pred

        latents = latents + (t_prev - t_curr) * pred

    # convert sequences of patches into img-like latents
    latents = unpack_latents(latents, latent_height, latent_width)

    return latents



def save_image(
    name: str,
    output_dir: str,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    verbose = True,
):
    if verbose:
        print(f"Saving image to {output_dir}/{name}")
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, name)

    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;txt2img"
    exif_data[ExifTags.Base.Make] = "Ovis"
    exif_data[ExifTags.Base.Model] = name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = prompt
    img.save(output_name, exif=exif_data, quality=95, subsampling=0)

