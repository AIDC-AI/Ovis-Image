<!-- <p align="center">
<img src=https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/3IK823BZ8w-mz_QfeYkDn.png width="30%"/>
</p> -->
<p align="center">
  <img src="docs/imgs/ovis_image_title.png" width="40%">
</p>

<!-- <h1 align="center">
Ovis-Image
</h1> -->

<p align="center">
  <!-- <a href="https://arxiv.org/abs/2506.23044"><img src="https://img.shields.io/badge/arXiv_paper-2506.23044-b31b1b.svg" alt="arxiv"></a> -->
  <a href="https://github.com/AIDC-AI/Ovis-Image/blob/main/docs/Ovis_Image_Technical_Report.pdf"><img src="https://img.shields.io/badge/Paper-Tech_Report-b31b1b" alt="paper"></a>
  <a href="https://github.com/AIDC-AI/Ovis-Image"><img src="https://img.shields.io/badge/GitHub-AIDC--AI/Ovis--Image-blue?style=flat&logo=github" alt="code"></a>
  <a href="https://huggingface.co/spaces/AIDC-AI/Ovis-Image-7B"><img src="https://img.shields.io/badge/🎨_HF_Spaces-AIDC--AI/Ovis--Image--7B-lightblack" alt="demo"></a>
  <a href="https://huggingface.co/AIDC-AI/Ovis-Image-7B"><img src="https://img.shields.io/badge/🤗_Model-AIDC--AI/Ovis--Image--7B-yellow" alt="model"></a>
</p>


Built upon [Ovis-U1](https://github.com/AIDC-AI/Ovis-U1), Ovis-Image is a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints. 


<p align="center">
  <img src="docs/imgs/ovis_image_arch.png" width="95%">
  <br>
  <em>The overall architecture of Ovis-Image (cf. Fig.2 in our report).</em>
</p>

## 🏆 Highlights

*   **Strong text rendering at a compact 7B scale**: Ovis-Image is a 7B text-to-image model that delivers text rendering quality comparable to much larger 20B-class systems such as Qwen-Image and competitive with leading closed-source models like GPT4o in text-centric scenarios, while remaining small enough to run on widely accessible hardware.
*   **High fidelity on text-heavy, layout-sensitive prompts**: The model excels on prompts that demand tight alignment between linguistic content and rendered typography (e.g., posters, banners, logos, UI mockups, infographics), producing legible, correctly spelled, and semantically consistent text across diverse fonts, sizes, and aspect ratios without compromising overall visual quality.
*   **Efficiency and deployability**: With its 7B parameter budget and streamlined architecture, Ovis-Image fits on a single high-end GPU with moderate memory, supports low-latency interactive use, and scales to batch production serving, bringing near–frontier text rendering to applications where tens-of-billions–parameter models are impractical.


## ✨ Showcase

Here are some examples demonstrating the capabilities of Ovis-Image.

<figure>
  <img src="docs/imgs/ovis_image_case.png" alt="Ovis-Image examples">
  <figcaption style="text-align: center;"></figcaption>
</figure>


## 🚀 News

- [2025/11/29] 🔥 Announcing Ovis-Image ([Model](https://huggingface.co/AIDC-AI/Ovis-Image-7B))!

## 🛠️ Inference

### Inference with Diffusers

First, install the `diffusers` library with support for Ovis-Image.

```bash
pip install git+https://github.com/DoctorKey/diffusers.git@ovis-image
```

Next, use the `OvisImagePipeline` to generate the image.

```python
import torch
from diffusers import OvisImagePipeline

pipe = OvisImagePipeline.from_pretrained("AIDC-AI/Ovis-Image-7B", torch_dtype=torch.bfloat16)
pipe.to("cuda")
prompt = "A creative 3D artistic render where the text \"OVIS-IMAGE\" is written in a bold, expressive handwritten brush style using thick, wet oil paint. The paint is a mix of vibrant rainbow colors (red, blue, yellow) swirling together like toothpaste or impasto art. You can see the ridges of the brush bristles and the glossy, wet texture of the paint. The background is a clean artist's canvas. Dynamic lighting creates soft shadows behind the floating paint strokes. Colorful, expressive, tactile texture, 4k detail."
image = pipe(prompt, negative_prompt="", num_inference_steps=50, true_cfg_scale=5.0).images[0]
image.save("ovis_image.png")
```

### Inference with Pytorch

Ovis-Image has been tested with Python 3.10, Torch 2.6.0, and Transformers 4.57.1. For a full list of package dependencies, please see `requirements.txt`.

```bash
git clone git@github.com:AIDC-AI/Ovis-Image.git
conda create -n ovis-image python=3.10 -y
conda activate ovis-image
cd Ovis-Image
pip install -r requirements.txt
pip install -e .
```

For text-to-image, please run

```bash
python ovis_image/test.py \
    --model_path AIDC-AI/Ovis-Image-7B/ovis_image.safetensors \
    --vae_path AIDC-AI/Ovis-Image-7B/ae.safetensors \
    --ovis_path AIDC-AI/Ovis-Image-7B/Ovis2.5-2B \
    --image_size 1024 \
    --denoising_steps 50 \
    --cfg_scale 5.0 \
    --prompt "A creative 3D artistic render where the text \"OVIS-IMAGE\" is written in a bold, expressive handwritten brush style using thick, wet oil paint. The paint is a mix of vibrant rainbow colors (red, blue, yellow) swirling together like toothpaste or impasto art. You can see the ridges of the brush bristles and the glossy, wet texture of the paint. The background is a clean artist's canvas. Dynamic lighting creates soft shadows behind the floating paint strokes. Colorful, expressive, tactile texture, 4k detail." \
```

Alternatively, you can try Ovis-Image directly in your browser on [![Hugging Face Space](https://img.shields.io/badge/🎨_HF_Spaces-AIDC--AI/Ovis--Image--7B-lightblack)](https://huggingface.co/spaces/AIDC-AI/Ovis-Image-7B)


## 📊 Performance


**Evaluation of text rendering ability on CVTG-2K.**

| Model | #Params. | WA (2 regions) | WA (3 regions) | WA (4 regions) | WA (5 regions) | WA (average) | NED↑ | CLIPScore↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seedream 3.0 | - | 0.6282 | 0.5962 | 0.6043 | 0.5610 | 0.5924 | 0.8537 | 0.7821 |
| GPT4o | - | 0.8779 | 0.8659 | 0.8731 | 0.8218 | 0.8569 | 0.9478 | 0.7982 |
| SD3.5 Large | 11B+8B | 0.7293 | 0.6825 | 0.6574 | 0.5940 | 0.6548 | 0.8470 | 0.7797 |
| RAG-Diffusion | 11B+12B | 0.4388 | 0.3316 | 0.2116 | 0.1910 | 0.2648 | 0.4498 | 0.7797 |
| FLUX.1-dev | 11B+12B | 0.6089 | 0.5531 | 0.4661 | 0.4316 | 0.4965 | 0.6879 | 0.7401 |
| TextCrafter | 11B+12B | 0.7628 | 0.7628 | 0.7406 | 0.6977 | 0.7370 | 0.8679 | 0.7868 |
| Qwen-Image | 7B+20B | 0.8370 | 0.8364 | 0.8313 | 0.8158 | 0.8288 | 0.9116 | 0.8017 |
| Ovis-Image | 2B+7B | **0.9248** | **0.9239** | **0.9180** | **0.9166** | **0.9200** | **0.9695** | **0.8368** |


**Evaluation of text rendering ability on LongText-Bench.**

| Model | #Params. | LongText-Bench-EN | LongText-Bench-ZN |
| :--- | :---: | :---: | :---: |
| Kolors 2.0 | - | 0.258 | 0.329 |
| GPT4o | - | **0.956** | 0.619 |
| Seedream 3.0 | - | 0.896 | 0.878 |
| OmniGen2 | 3B+4B | 0.561 | 0.059 |
| Janus-Pro | 7B | 0.019 | 0.006 |
| BLIP3-o | 7B+1B | 0.021 | 0.018 |
| FLUX.1-dev | 11B+12B | 0.607 | 0.005 |
| BAGEL | 7B+7B | 0.373 | 0.310 |
| HiDream-I1-Full | 11B+17B | 0.543 | 0.024 |
| Qwen-Image | 7B+20B | 0.943 | 0.946 |
| Ovis-Image | 2B+7B | 0.922 | **0.964** |


**Evaluation of text-to-image generation ability on DPG-Bench.**

| Model | #Params. | Global | Entity | Attribute | Relation | Other | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seedream 3.0 | - | **94.31** | **92.65** | 91.36 | 92.78 | 88.24 | 88.27 |
| GPT4o | - | 88.89 | 88.94 | 89.84 | 92.63 | 90.96 | 85.15 |
| Ovis-U1 | 2B+1B | 82.37 | 90.08 | 88.68 | 93.35 | 85.20 | 83.72 |
| OmniGen2 | 3B+4B | 88.81 | 88.83 | 90.18 | 89.37 | 90.27 | 83.57 |
| Janus-Pro | 7B | 86.90 | 88.90 | 89.40 | 89.32 | 89.48 | 84.19 |
| BAGEL | 7B+7B | 88.94 | 90.37 | 91.29 | 90.82 | 88.67 | 85.07 |
| HiDream-I1-Full | 11B+17B | 76.44 | 90.22 | 89.48 | 93.74 | 91.83 | 85.89 |
| UniWorld-V1 | 7B+12B | 83.64 | 88.39 | 88.44 | 89.27 | 87.22 | 81.38 |
| Qwen-Image | 7B+20B | 91.32 | 91.56 | **92.02** | **94.31** | **92.73** | **88.32** |
| Ovis-Image | 2B+7B | 82.37 | 92.38 | 90.42 | 93.98 | 91.20 | 86.59 |


**Evaluation of text-to-image generation ability on GenEval.**

| Model | #Params. | Single object | Two object | Counting | Colors | Position | Attribute binding | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seedream 3.0 | - | 0.99 | 0.96 | **0.91** | **0.93** | 0.47 | **0.80** | 0.84 |
| GPT4o | - | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| Ovis-U1 | 2B+1B | 0.98 | **0.98** | 0.90 | 0.92 | **0.79** | 0.75 | **0.89** |
| OmniGen2 | 3B+4B | **1.00** | 0.95 | 0.64 | 0.88 | 0.55 | 0.76 | 0.80 |
| Janus-Pro | 7B | 0.99 | 0.89 | 0.59 | 0.90 | **0.79** | 0.66 | 0.80 |
| BAGEL | 7B+7B | 0.99 | 0.94 | 0.81 | 0.88 | 0.64 | 0.63 | 0.82 |
| HiDream-I1-Full | 11B+17B | 1.00 | **0.98** | 0.79 | 0.91 | 0.60 | 0.72 | 0.83 |
| UniWorld-V1 | 7B+12B | 0.99 | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 | 0.80 |
| Qwen-Image | 7B+20B | 0.99 | 0.92 | 0.89 | 0.88 | 0.76 | 0.77 | 0.87 |
| Ovis-Image | 2B+7B | **1.00** | 0.97 | 0.76 | 0.86 | 0.67 | **0.80** | 0.84 |



**Evaluation of text-to-image generation ability on OneIG-EN.**

| Model | #Params. | Alignment | Text | Reasoning | Style | Diversity | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Kolors 2.0 | - | 0.820 | 0.427 | 0.262 | 0.360 | 0.300 | 0.434 |
| Imagen4 | - | 0.857 | 0.805 | 0.338 | 0.377 | 0.199 | 0.515 |
| Seedream 3.0 | - | 0.818 | 0.865 | 0.275 | 0.413 | 0.277 | 0.530 |
| GPT4o | - | 0.851 | 0.857 | **0.345** | **0.462** | 0.151 | 0.533 |
| Ovis-U1 | 2B+1B | 0.816 | 0.034 | 0.226 | 0.443 | 0.191 | 0.342 |
| CogView4 | 6B | 0.786 | 0.641 | 0.246 | 0.353 | 0.205 | 0.446 |
| Janus-Pro | 7B | 0.553 | 0.001 | 0.139 | 0.276 | **0.365** | 0.267 |
| OmniGen2 | 3B+4B | 0.804 | 0.680 | 0.271 | 0.377 | 0.242 | 0.475 |
| BLIP3-o | 7B+1B | 0.711 | 0.013 | 0.223 | 0.361 | 0.229 | 0.307 |
| FLUX.1-dev | 11B+12B | 0.786 | 0.523 | 0.253 | 0.368 | 0.238 | 0.434 |
| BAGEL | 7B+7B | 0.769 | 0.244 | 0.173 | 0.367 | 0.251 | 0.361 |
| BAGEL+CoT | 7B+7B | 0.793 | 0.020 | 0.206 | 0.390 | 0.209 | 0.324 |
| HiDream-I1-Full | 11B+17B | 0.829 | 0.707 | 0.317 | 0.347 | 0.186 | 0.477 |
| HunyuanImage-2.1 | 7B+17B | 0.835 | 0.816 | 0.299 | 0.355 | 0.127 | 0.486 |
| Qwen-Image | 7B+20B | **0.882** | 0.891 | 0.306 | 0.418 | 0.197 | **0.539** |
| Ovis-Image | 2B+7B | 0.858 | **0.914** | 0.308 | 0.386 | 0.186 | 0.530 |



**Evaluation of text-to-image generation ability on OneIG-ZN.**

| Model | #Params. | Alignment | Text | Reasoning | Style | Diversity | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Kolors 2.0 | - | 0.738 | 0.502 | 0.226 | 0.331 | 0.333 | 0.426 |
| Seedream 3.0 | - | 0.793 | 0.928 | 0.281 | 0.397 | 0.243 | 0.528 |
| GPT4o | - | 0.812 | 0.650 | **0.300** | **0.449** | 0.159 | 0.474 |
| CogView4 | 6B | 0.700 | 0.193 | 0.236 | 0.348 | 0.214 | 0.338 |
| Janus-Pro | 7B | 0.324 | 0.148 | 0.104 | 0.264 | **0.358** | 0.240 |
| BLIP3-o | 7B+1B | 0.608 | 0.092 | 0.213 | 0.369 | 0.233 | 0.303 |
| BAGEL | 7B+7B | 0.672 | 0.365 | 0.186 | 0.357 | 0.268 | 0.370 |
| BAGEL+CoT | 7B+7B | 0.719 | 0.127 | 0.219 | 0.385 | 0.197 | 0.329 |
| HiDream-I1-Full | 11B+17B | 0.620 | 0.205 | 0.256 | 0.304 | 0.300 | 0.337 |
| HunyuanImage-2.1 | 7B+17B | 0.775 | 0.896 | 0.271 | 0.348 | 0.114 | 0.481 |
| Qwen-Image | 7B+20B | **0.825** | **0.963** | 0.267 | 0.405 | 0.279 | **0.548** |
| Ovis-Image | 2B+7B | 0.805 | 0.961 | 0.273 | 0.368 | 0.198 | 0.521 |

## 📚 Citation

If you find Ovis-Image useful for your research or applications, please cite our technical report:

```bibtex
@misc{wang2025ovis_image,
  title={Ovis-Image Technical Report}, 
  author={Wang, Guo-Hua and Cao, Liangfu and Cui, Tianyu and Fu, Minghao and Chen, Xiaohao and Zhan, Pengxin and Zhao, Jianshan and Li, Lan and Fu, Bowen and Liu, Jiaqi and Chen, Qing-Guo},
  howpublished={\url{https://github.com/AIDC-AI/Ovis-Image}},
  year={2025}
}
```

## 🙏 Acknowledgments

The code is built upon [Ovis](https://github.com/AIDC-AI/Ovis) and [FLUX](https://github.com/black-forest-labs/flux). We thank their authors for open-sourcing their great work.

## 📄 License

This project is licensed under the Apache License, Version 2.0 (SPDX-License-Identifier: Apache-2.0). 

## 🚨 Disclaimer

We used compliance checking algorithms during the training process, to ensure the compliance of the trained model to the best of our ability. Due to complex data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.


## 🔥 We are hiring!

We are looking for both interns and full-time researchers to join our team, focusing on multimodal understanding, generation, reasoning, AI agents, and unified multimodal models. If you are interested in exploring these exciting areas, please reach out to us at qingguo.cqg@alibaba-inc.com.
