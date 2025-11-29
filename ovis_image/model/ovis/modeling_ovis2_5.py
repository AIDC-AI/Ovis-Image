# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
import math
from typing import Dict, List, Optional, Tuple, Union

import PIL.Image
import numpy as np
import torch
# from flash_attn import flash_attn_varlen_func
# from flash_attn.layers.rotary import apply_rotary_emb
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel

from ovis_image.model.ovis.configuration_ovis2_5 import Siglip2NavitConfig, Ovis2_5_Config
from ovis_image.model.ops import get_attention_type_by_system

IMAGE_PLACEHOLDER = "<image>"
IMAGE_PLACEHOLDER_ID = -200
VIDEO_PLACEHOLDER = "<video>"
VIDEO_PLACEHOLDER_ID = -201

VISUAL_ATOM_ID = -300
INDICATOR_IDS = [-301, -302, -303, -304]

# copied from qwen2.5-vl
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = config.num_patches
        self.preserve_original_pe = config.preserve_original_pe
        self.hidden_stride = config.hidden_stride


        # siglip2 naflex
        if self.num_patches > 0:
            self.patch_embedding = nn.Linear(
                    in_features=config.num_channels * self.patch_size * self.patch_size,
                    out_features=self.embed_dim,
                )
            if self.preserve_original_pe:
                self.position_embedding_size = int(self.num_patches**0.5)
                self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

        else:
            self.patch_embedding = nn.Conv2d(
                    in_channels=config.num_channels,
                    out_channels=self.embed_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    padding="valid",
                )
            if self.preserve_original_pe:
                self.num_patches = (self.image_size // self.patch_size) ** 2
                self.position_embedding_size = self.image_size // self.patch_size
                self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        
    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, 
                grid_thws: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (num_patches, num_channels * temporal_patch_size * patch_size * patch_size)
            grid_thws: (`torch.LongTensor`):
                grid shape (num_patches, 3)
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        if isinstance(self.patch_embedding, nn.Linear):
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        elif isinstance(self.patch_embedding, nn.Conv2d):
            pixel_values = pixel_values.view(-1, self.config.num_channels * self.config.temporal_patch_size, self.patch_size,
                   self.patch_size)
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.reshape(-1, self.embed_dim)


        if self.preserve_original_pe:
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(patch_embeds)
            ori_h = ori_w = self.position_embedding_size
            positional_embeddings = self.position_embedding.weight.reshape(
                                self.position_embedding_size, self.position_embedding_size, -1
                            ).unsqueeze(0).permute(0,3,1,2)
            # pos_embed = self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2)
            cnt = 0
            for t, h, w in grid_thws:
                thw = t * h * w
                pe = F.interpolate(positional_embeddings, size=(h, w), mode='bicubic', align_corners=False)
                pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
                pe = pe[0].repeat(t, 1)
                pe = pe.reshape(t, h // self.hidden_stride, self.hidden_stride, w // self.hidden_stride,
                                self.hidden_stride, -1)
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(thw, -1)
                pos_embed_new[cnt:cnt + thw] = pe
                cnt += thw
            patch_embeds = patch_embeds + pos_embed_new

        return patch_embeds



def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_flashatt(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_rope = config.use_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim)
        keys = keys.view(seq_length, self.num_heads, self.head_dim)
        values = values.view(seq_length, self.num_heads, self.head_dim)

        if self.use_rope:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb_flashatt(queries.unsqueeze(0), keys.unsqueeze(0), cos, sin)
            queries = queries.squeeze(0)
            keys = keys.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        # attn_output = flash_attn_varlen_func(queries, keys, values, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        #                                     seq_length, -1
        #                                 )
        batch_size = cu_seqlens.shape[0] - 1
        q_list, k_list, v_list = [], [], []
        for i in range(batch_size):
            start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
            q_list.append(queries[start:end])
            k_list.append(keys[start:end])
            v_list.append(values[start:end])
        
        def pad_to_max(t, max_len):
            pad = (0, 0, 0, 0, 0, max_len - t.shape[0])  # [seqlen, num_heads, head_dim]
            return torch.nn.functional.pad(t, pad)
        
        q_batched = torch.stack([pad_to_max(x, max_seqlen) for x in q_list])  # (batch, seqlen, nhead, dim)
        k_batched = torch.stack([pad_to_max(x, max_seqlen) for x in k_list])
        v_batched = torch.stack([pad_to_max(x, max_seqlen) for x in v_list])
        
        mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=queries.device)
        for i in range(batch_size):
            mask[i, :q_list[i].shape[0]] = True  # [batch, seqlen]
        
        # (batch, nhead, seqlen, head_dim)
        q_batched = q_batched.transpose(1, 2)  # (batch, nhead, seqlen, head_dim)
        k_batched = k_batched.transpose(1, 2)
        v_batched = v_batched.transpose(1, 2)
        if torch.__version__ >= "2.7.0":
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q_batched, k_batched, v_batched,
                    attn_mask=mask.unsqueeze(1).unsqueeze(2)
                ).permute(0, 2, 1, 3).reshape(seq_length, -1)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_batched, k_batched, v_batched,
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # broadcast到 (batch, 1, 1, seqlen)
            ).permute(0, 2, 1, 3).reshape(seq_length, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output

class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor, 
        position_embeddings: torch.Tensor
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens, 
            position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2NavitConfig
    """

    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.rotary_pos_emb = VisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        self.patch_size = config.patch_size
        self.hidden_stride = config.hidden_stride
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        self.fullatt_block_indexes = None if config.fullatt_block_indexes is None else [int(i) for i in config.fullatt_block_indexes.split('|')]
        

    # copied from qwen2.5_vl
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.hidden_stride // self.patch_size  # patch (after merge) number in each window

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.hidden_stride,  # number of patch after merge
                grid_w // self.hidden_stride,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        grid_thws: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=inputs_embeds.device,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = inputs_embeds.size()
        inputs_embeds = inputs_embeds.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        inputs_embeds = inputs_embeds[window_index, :, :]
        inputs_embeds = inputs_embeds.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        reverse_indices = torch.argsort(window_index)
        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for index, block in enumerate(self.layers):
            if self.fullatt_block_indexes is None or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(block.__call__, hidden_states, cu_seqlens_tmp, position_embeddings)
            else:
                hidden_states = block(hidden_states, cu_seqlens_tmp, position_embeddings)
            if output_hidden_states:
                hidden_states_ = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                encoder_states += (hidden_states_[reverse_indices, :].reshape(seq_len, -1),)
        # tokens = self.post_trunk_norm(tokens)
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :].reshape(seq_len, -1)

        return hidden_states, encoder_states

class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        """
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        hidden_states = self.embeddings(pixel_values, grid_thws)

        last_hidden_state, hidden_states = self.encoder(hidden_states, grid_thws, output_hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            output = (last_hidden_state,)
            output += (hidden_states,) if output_hidden_states else ()
            return output
        
        return BaseModelOutputWithNoAttention(
                last_hidden_state=last_hidden_state, 
                hidden_states=hidden_states
            )
       
class Siglip2PreTrainedModel(PreTrainedModel):
    config_class = Siglip2NavitConfig
    base_model_prefix = "siglip2_navit"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = True


class Siglip2NavitModel(Siglip2PreTrainedModel):
    config_class = Siglip2NavitConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2NavitConfig):
        super().__init__(config)

        self.vision_model = Siglip2VisionTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:
        
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class VisualEmbedding(torch.nn.Embedding):
    """
    A visual embedding layer that can handle both discrete token IDs (long) and continuous
    soft-token probabilities (float).
    """

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        # Handle soft tokens (probabilities) by matrix multiplication with the embedding weight
        return torch.matmul(visual_tokens, self.weight)


class VisualTokenizer(torch.nn.Module):
    """
    Tokenizes images or videos into a sequence of continuous visual tokens.
    """

    def __init__(self, vit, visual_vocab_size, image_processor_name_or_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit = vit
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name_or_path, do_center_crop=False)
        head_dim = visual_vocab_size - len(INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.vit.config.hidden_size * self.vit.config.hidden_stride ** 2, head_dim, bias=False),
            torch.nn.LayerNorm(head_dim)
        )

    def _encode(self, pixel_values, grid_thws):
        output = self.vit(pixel_values, grid_thws, output_hidden_states=True, return_dict=True)
        features = output.hidden_states[-1]
        seq_len, _ = features.shape
        features = features.reshape(seq_len // (self.vit.config.hidden_stride ** 2), -1)
        return features

    # Adapted from qwen2_vl
    @staticmethod
    def smart_resize(
        height: int, width: int, factor: int = 28, min_pixels: int = 448 * 448, max_pixels: int = 1344 * 1792
    ):
        """Rescales the image so that the following conditions are met:
        1. Both dimensions are divisible by 'factor'.
        2. The total number of pixels is within ['min_pixels', 'max_pixels'].
        3. The aspect ratio is maintained as closely as possible.
        """
        if height < factor or width < factor:
            if height < width:
                width = round(factor / height * width)
                height = factor
            else:
                height = round(factor / width * height)
                width = factor

        elif max(height, width) / min(height, width) > 200:
            if height > width:
                height = 200 * width
            else:
                width = 200 * height

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def preprocess(
        self,
        image: Optional[PIL.Image.Image] = None,
        video: Optional[List[PIL.Image.Image]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None
    ):
        patch_size = self.vit.config.patch_size
        temporal_patch_size = self.vit.config.temporal_patch_size
        hidden_stride = self.vit.config.hidden_stride
        assert (image is None) ^ (video is None), "Invalid input: expect either image or video"
        if image is not None:
            images = [image]
        else:
            images = video
        images = [image.convert("RGB") if image.mode != 'RGB' else image for image in images]
        width, height = images[0].size
        processed_images = []
        for image in images:
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=patch_size * hidden_stride,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            new_size = dict(height=resized_height, width=resized_width)
            new_image = self.image_processor.preprocess(image, size=new_size, return_tensors="np")['pixel_values'][0]
            processed_images.append(new_image)

        patches = np.array(processed_images)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])

        patches = patches.reshape(
            grid_t, temporal_patch_size, channel,
            grid_h // hidden_stride, hidden_stride, patch_size,
            grid_w // hidden_stride, hidden_stride, patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        flatten_patches = torch.tensor(flatten_patches)

        return flatten_patches, grid_thw

    def forward(
        self, pixel_values, grid_thws
    ) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self._encode(pixel_values, grid_thws)
        logits = self.head(features)
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

        token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(token_len, len(INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=1)
        return tokens


class OvisPreTrainedModel(PreTrainedModel):
    config_class = Ovis2_5_Config
    base_model_prefix = "ovis2_5"


class Ovis2_5(OvisPreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True

    def __init__(self, config: Ovis2_5_Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        attn_implementation = get_attention_type_by_system()
        print(f"Use {attn_implementation} for LLM!")
        self.llm = AutoModelForCausalLM.from_config(
            self.config.llm_config,
            attn_implementation=attn_implementation,
        )
        assert self.config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
        self.visual_tokenizer = VisualTokenizer(vit=AutoModel.from_config(self.config.vit_config),
                                                visual_vocab_size=self.config.visual_vocab_size,
                                                image_processor_name_or_path=self.config.name_or_path)

        self.vte = VisualEmbedding(self.config.visual_vocab_size, self.config.hidden_size,
                                   device=self.visual_tokenizer.vit.device, dtype=self.visual_tokenizer.vit.dtype)
        indicator_token_indices = torch.arange(
            self.config.visual_vocab_size - len(INDICATOR_IDS),
            self.config.visual_vocab_size,
            dtype=torch.long
        )
        self.register_buffer("indicator_token_indices", indicator_token_indices, persistent=False)

        def _merge_modules(modules_list: tuple):
            merged_modules = []
            for modules in modules_list:
                merged_modules.extend(modules if modules else [])
            return merged_modules

        # Standard model configurations for parallelism and device placement
        self._no_split_modules = _merge_modules(
            (self.llm._no_split_modules, self.visual_tokenizer.vit._no_split_modules))
        self._skip_keys_device_placement = self.llm._skip_keys_device_placement
        self._keep_in_fp32_modules = _merge_modules(
            (self.llm._keep_in_fp32_modules, self.visual_tokenizer.vit._keep_in_fp32_modules))
        self.is_parallelizable = all((self.llm.is_parallelizable, self.visual_tokenizer.vit.is_parallelizable))
        # self.supports_gradient_checkpointing = True
        self.supports_gradient_checkpointing = False

    def tie_weights(self):
        self.llm.tie_weights()

    def get_wte(self):
        return self.llm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        grid_thws: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        inputs_embeds = self.merge_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def merge_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        grid_thws: Optional[torch.Tensor],
    ):
        placeholder_token_mask = torch.lt(input_ids, 0)
        multimodal_embeds = self.get_wte()(torch.masked_fill(input_ids, placeholder_token_mask, 0))

        if pixel_values is not None:
            visual_indicator_embeds = self.vte(self.indicator_token_indices).to(
                dtype=multimodal_embeds.dtype, device=multimodal_embeds.device
            )
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(dtype=multimodal_embeds.dtype, device=multimodal_embeds.device)

            for i, indicator_id in enumerate(INDICATOR_IDS):
                multimodal_embeds[input_ids == indicator_id] = visual_indicator_embeds[i]
            multimodal_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds

        return multimodal_embeds

    def _merge_inputs(
        self, raw_input_ids, placeholder_id, grid_thws, indicator_begin_id, indicator_end_id
    ):
        input_ids = []
        prev_index = 0
        placeholder_indexes = [i for i, v in enumerate(raw_input_ids) if v == placeholder_id]
        for placeholder_index, grid_thw in zip(placeholder_indexes, grid_thws):
            input_ids.extend(raw_input_ids[prev_index:placeholder_index])
            num_image_atoms = grid_thw.prod().item()
            num_image_atoms //= self.visual_tokenizer.vit.config.hidden_stride ** 2
            num_image_atoms //= self.visual_tokenizer.vit.config.temporal_patch_size
            input_ids.extend([indicator_begin_id] + [VISUAL_ATOM_ID] * num_image_atoms + [indicator_end_id])
            prev_index = placeholder_index + 1
        input_ids.extend(raw_input_ids[prev_index:])
        return input_ids

    def _tokenize_with_visual_placeholder(self, text):
        placeholder = VIDEO_PLACEHOLDER if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER
        placeholder_id = VIDEO_PLACEHOLDER_ID if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER_ID
        chunks = [self.text_tokenizer(chunk, add_special_tokens=False).input_ids for chunk in text.split(placeholder)]
        input_ids = chunks[0]
        for chunk in chunks[1:]:
            input_ids.append(placeholder_id)
            input_ids.extend(chunk)
        return input_ids

    def preprocess_inputs(
        self,
        messages: List[Union[str, Dict]],
        min_pixels=448 * 448,
        max_pixels=1344 * 1792,
        add_generation_prompt=True,
        enable_thinking=False
    ):
        text = self.text_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )
        input_ids = self._tokenize_with_visual_placeholder(text)
        images = []
        videos = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                images.extend([item["image"] for item in content if item.get("image") is not None])
                videos.extend([item["video"] for item in content if item.get("video") is not None])
        if images and videos:
            raise ValueError(
                "Multiple visual input data types detected (both image and video provided). "
                "This model supports only one type of visual input data at a time. "
                "Please provide either image or video, but not both."
            )

        pixel_values, grid_thws = None, None
        if images:
            pixel_values, grid_thws = zip(
                *(self.visual_tokenizer.preprocess(image=image, min_pixels=min_pixels, max_pixels=max_pixels)
                  for image in images)
            )
            input_ids = self._merge_inputs(
                input_ids, IMAGE_PLACEHOLDER_ID, grid_thws, INDICATOR_IDS[0], INDICATOR_IDS[1]
            )
            pixel_values = torch.cat(pixel_values, dim=0)
            grid_thws = torch.cat(grid_thws, dim=0)
        elif videos:
            assert len(videos) == 1, "only support single video"
            pixel_values, grid_thws = self.visual_tokenizer.preprocess(
                video=videos[0], min_pixels=min_pixels, max_pixels=max_pixels
            )
            input_ids = self._merge_inputs(
                input_ids, VIDEO_PLACEHOLDER_ID, grid_thws, INDICATOR_IDS[2], INDICATOR_IDS[3]
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        return input_ids, pixel_values, grid_thws

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        attention_mask = torch.ne(inputs, self.text_tokenizer.pad_token_id).to(device=inputs.device)
        inputs_embeds = self.merge_multimodal(
            input_ids=inputs,
            pixel_values=kwargs.pop('pixel_values', None),
            grid_thws=kwargs.pop('grid_thws', None)
        )
        enable_thinking = kwargs.pop('enable_thinking', False)
        enable_thinking_budget = kwargs.pop('enable_thinking_budget', False)
        thinking_budget = kwargs.pop('thinking_budget', 1024)
        
        if enable_thinking and enable_thinking_budget:
            actual_max_new_tokens = kwargs['max_new_tokens']
            kwargs['max_new_tokens'] = thinking_budget
            generated_ids = self.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
            output_ids = generated_ids
            output_ids_list = generated_ids[0]
            
            # check if the generation has already finished (151645 is <|im_end|>)
            if 151645 not in output_ids_list:
                # check if the thinking process has finished (151668 is </think>)
                # and prepare the second model input
                if 151668 not in output_ids_list:
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    early_stopping_ids = self.text_tokenizer(early_stopping_text, return_tensors="pt", return_attention_mask=False).input_ids.to(inputs.device)
                    input_ids_appendent = torch.cat([output_ids, early_stopping_ids], dim=-1)
                    kwargs['streamer'].put(early_stopping_ids) if 'streamer' in kwargs else None
                else:
                    input_ids_appendent = output_ids
                

                # second generation
                new_inputs = torch.cat([inputs, input_ids_appendent], dim=-1)
                attention_mask = torch.ne(new_inputs, self.text_tokenizer.pad_token_id).to(device=inputs.device)
                inputs_embeds_appendent = self.merge_multimodal(
                    input_ids=input_ids_appendent,
                    pixel_values=None,
                    grid_thws=None
                )
                new_inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_appendent], dim=-2)
                
                kwargs['max_new_tokens'] = inputs_embeds.size(-2) + actual_max_new_tokens - new_inputs_embeds.size(-2)
                generated_ids2 = self.llm.generate(inputs=None, inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, **kwargs)
                kwargs['streamer'].manual_end() if 'streamer' in kwargs else None
                return torch.cat([input_ids_appendent, generated_ids2], dim=-1)
            
            else:
                kwargs['streamer'].manual_end() if 'streamer' in kwargs else None
                return generated_ids
            
        else:
            generated_ids = self.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
            kwargs['streamer'].manual_end() if 'streamer' in kwargs else None
            return generated_ids


AutoConfig.register('siglip2_navit', Siglip2NavitConfig)
AutoModel.register(Siglip2NavitConfig, Siglip2NavitModel)
AutoConfig.register("ovis2_5", Ovis2_5_Config)
AutoModelForCausalLM.register(Ovis2_5_Config, Ovis2_5)
