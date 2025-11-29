# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
import math
import torch
import torchvision
from torchvision import transforms
from einops import rearrange, repeat


def ceil_to(x, factor=16):
    return math.ceil(float(x) / factor) * factor


def build_img_ids(
    latent_height,
    latent_width,
    latent_crop_height = None,
    latent_crop_width = None,
    time = 0,
):
    if latent_crop_height is None:
        latent_crop_height = latent_height
    if latent_crop_width is None:
        latent_crop_width = latent_width
    img_ids = torch.zeros(latent_height, latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(latent_width)[None, :]
    # crop
    crop_h = (latent_height - latent_crop_height) // 2
    crop_w = (latent_width - latent_crop_width) // 2
    img_ids = img_ids[crop_h:crop_h+latent_crop_height, crop_w:crop_w+latent_crop_width]
    img_ids[..., 0] = time
    h, w, c = img_ids.shape
    img_ids = img_ids.reshape(h * w, c)
    return img_ids


def process_pil_img_to_tensor(
    pil_img,
    output_size: int | None = 256,
    output_width: int | None = None,
    output_height: int | None = None,
    with_position_ids: bool = False,
    position_ids_time: int = 0,
):
    width, height = pil_img.size
    if output_width is None or output_height is None:
        output_width = output_size
        output_height = output_size
    assert output_height % 16 == 0
    assert output_width % 16 == 0
    resize_ratio = max(
        float(output_width)/width, 
        float(output_height)/height
    )
    resize_size = (
        ceil_to(resize_ratio * height, 16), 
        ceil_to(resize_ratio * width, 16)
    )
    pil_resize_img = torchvision.transforms.functional.resize(
        pil_img, resize_size, interpolation=transforms.InterpolationMode.BICUBIC
    )
    pil_crop_img = torchvision.transforms.functional.center_crop(
        pil_resize_img, (output_height, output_width)
    )
    image_tensor = torchvision.transforms.functional.to_tensor(pil_crop_img)
    image_tensor = torchvision.transforms.functional.normalize(
        image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    if with_position_ids:
        img_ids = build_img_ids(
            latent_height = resize_size[0] // 16,
            latent_width = resize_size[1] // 16,
            latent_crop_height = output_height // 16,
            latent_crop_width = output_width // 16,
            time = position_ids_time,
        )
    else:
        img_ids = None
    return pil_crop_img, image_tensor, img_ids


def pack_latent_to_token(
    latent,
):
    token = rearrange(
        latent, 
        "b c (h ph) (w pw) -> b (h w) (c ph pw)", 
        ph=2, 
        pw=2
    )
    return token


def unpack_token_to_latent(
    token, 
    image_height: int | None = None,
    latent_height: int | None  = None,
    image_width: int | None  = None, 
    latent_width: int | None  = None,
):
    if image_height is not None:
        h = math.ceil(image_height / 16)
    elif latent_height is not None:
        h = latent_height // 2
    else:
        raise ValueError(f"both {image_height} and {latent_height} are None")
    if image_width is not None:
        w = math.ceil(image_width / 16)
    elif latent_width is not None:
        w = latent_width // 2
    else:
        raise ValueError(f"both {image_width} and {latent_width} are None")
    return rearrange(
        token,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h,
        w=w,
        ph=2,
        pw=2,
    )


