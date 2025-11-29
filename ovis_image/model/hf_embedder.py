# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import torch
from torch import nn, Tensor

from ovis_image.model.ovis.modeling_ovis2_5 import Ovis2_5, Ovis2_5_Config


class OvisEmbedder(nn.Module):
    def __init__(
            self, 
            model_path: str, 
            random_init=False, 
            **hf_kwargs
        ):
        super().__init__()
        if random_init:
            # Initialize Ovis model with random weights for test purpose only
            config = Ovis2_5_Config.from_pretrained(model_path)
            config.name_or_path = model_path
            self.hf_module = Ovis2_5._from_config(config, **hf_kwargs)
        else:
            self.hf_module = Ovis2_5.from_pretrained(
                model_path, **hf_kwargs
            )
        self.pad_token_id = self.hf_module.text_tokenizer.pad_token_id
        self.user_prompt_begin_id = 28
        # get Qwen3
        self.hf_module = self.hf_module.llm.model
        self.hf_module = self.hf_module.eval().requires_grad_(False)


    def forward(self, batch_tokens: Tensor, attention_mask = None) -> Tensor:
        if attention_mask is None:
            attention_mask = torch.ne(
                batch_tokens, self.pad_token_id
            ).to(device=batch_tokens.device)
        outputs = self.hf_module(
            input_ids=batch_tokens,
            attention_mask=attention_mask,
        )
        txt_semantic_embed = outputs.last_hidden_state
        txt_semantic_embed = txt_semantic_embed * attention_mask[..., None]
        txt_semantic_embed = txt_semantic_embed[:, self.user_prompt_begin_id:, :]
        return txt_semantic_embed


