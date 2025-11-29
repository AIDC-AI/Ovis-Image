# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse

import torch
from safetensors.torch import load_file

from ovis_image.model.tokenizer import build_ovis_tokenizer
from ovis_image.model.autoencoder import load_ae
from ovis_image.model.hf_embedder import OvisEmbedder
from ovis_image.model.model import OvisImageModel
from ovis_image.sampling import generate_image, save_image
from ovis_image import ovis_image_configs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--ovis_path', type=str, default="")
    parser.add_argument('--vae_path', type=str, default="")
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--denoising_steps', type=int, default=50)
    parser.add_argument('--cfg_scale', type=float, default=5.0)
    args = parser.parse_args()
    return args

def load_model_weight(model, model_path):
    model_state_dict = load_file(model_path)
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict)
    print(f"Load Missing Keys {missing_keys}")
    print(f"Load Unexpected Keys {unexpected_keys}")
    return model


def main():
    args = parse_args()
    model_config = ovis_image_configs["ovis-image-7b"]
    device = "cuda"
    _dtype = torch.bfloat16
    print(f"dtype: {_dtype}")
    ovis_image = OvisImageModel(model_config)
    ovis_image = load_model_weight(ovis_image, args.model_path)
    ovis_image = ovis_image.to(device=device, dtype=_dtype)
    ovis_image.eval()

    ovis_tokenizer = build_ovis_tokenizer(args.ovis_path)
    autoencoder = load_ae(
        args.vae_path,
        model_config.autoencoder_params,
        device=device,
        dtype=_dtype,
        random_init=False,
    )
    autoencoder.eval()
    ovis_encoder = OvisEmbedder(
        model_path=args.ovis_path,
        random_init=False,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device=device, dtype=_dtype)

    with torch.no_grad():
        image = generate_image(
            device=device,
            dtype=_dtype,
            model=ovis_image,
            prompt=args.prompt,
            autoencoder=autoencoder,
            ovis_tokenizer=ovis_tokenizer,
            ovis_encoder=ovis_encoder,
            img_height=args.image_size,
            img_width=args.image_size,
            denoising_steps=args.denoising_steps,
            cfg_scale=args.cfg_scale,
            seed=42,
        )
    image_name = f"ovis_image.png"
    save_image(
        name=image_name,
        output_dir="outputs",
        x=image,
        add_sampling_metadata=True,
        prompt=args.prompt,
        verbose=False,
    )


if __name__ == "__main__":
    main()
