# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend
flash_attn_func = None
try:
    from flash_attn_interface import flash_attn_func
    print("find flash attn 3")
except:
    flash_attn_func = None


def check_attention_type(attn_implementation):
    if torch.__version__ >= "2.7.0":
        if attn_implementation != "sdpa":
            print("please set attn_implementation as sdpa for torch271")
    elif flash_attn_func is not None:
        if attn_implementation != "flash_attention_3":
            print("please set attn_implementation as flash_attention_3 for H100")

def get_attention_type_by_system():
    if torch.__version__ >= "2.7.0":
        return "sdpa"
    elif flash_attn_func is not None:
        return "flash_attention_3"
    else:
        return "eager"


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    if torch.__version__ >= "2.7.0":
        return attention_sdpa(q, k, v, pe)
    elif flash_attn_func is not None:
        return attention_fa3(q, k, v, pe)
    else:
        return attention_eager(q, k, v, pe)


def attention_eager(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def attention_sdpa(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # B200用torch271镜像，用SDPA加速
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
        )
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def attention_fa3(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # H100上用flash_attn_3加速
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    x = flash_attn_func(q, k, v)[0]
    x = rearrange(x, "B L H D -> B L (H D)")
    return x

def get_attention_func(attn_implementation):
    if attn_implementation == "eager":
        return attention_eager
    elif attn_implementation == "sdpa":
        return attention_sdpa
    elif attn_implementation == "flash_attention_3":
        return attention_fa3
    else:
        return attention_eager


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()
    

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

