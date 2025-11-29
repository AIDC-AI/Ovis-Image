# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from typing import List

import torch
from transformers import AutoTokenizer


class OvisTokenizer:
    """
    Tokenizing and encoding/decoding text using the Ovis tokenizer.

    Args:
        model_path (str): Path to the tokenzier from hugging face.

    """

    def __init__(
        self, 
        model_path: str = "Ovis2.5-2B", 
        max_length: int = 256, 
        **hf_kwargs
    ):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)
        self.system_prompt = "Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: "
        self.user_prompt_begin_id = 28
        self._max_length = max_length + self.user_prompt_begin_id

    def encode(
        self,
        s: str,
        system_prompt = ""
    ) -> torch.Tensor:
        """
        Encode the prompt text into tokens.
        """
        if len(system_prompt) == 0:
            system_prompt = self.system_prompt
        messages = [{
            "role": "user",
            "content": system_prompt + s,
        }]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        tokens = self._tokenizer(
            text, 
            padding="max_length", 
            truncation=True,
            max_length=self._max_length, 
            return_tensors="pt", 
            add_special_tokens=False,
        )
        return tokens.input_ids, tokens.attention_mask

    def decode(self, t: List[int]) -> str:
        return self._tokenizer.decode(t, skip_special_tokens=False)


def build_ovis_tokenizer(tokenizer_path, **hf_kwargs):
    max_ovis_encoding_len = 256
    ovis_tokenizer = OvisTokenizer(
        tokenizer_path, 
        max_length=max_ovis_encoding_len,
        **hf_kwargs,
    )
    return ovis_tokenizer


if __name__ == "__main__":
    ovis_path = "/mnt/workspace/cv_multimodal/aigc/huggingface/Ovis2.5-2B"
    text = "a cute cat"
    ovis_tokenizer = OvisTokenizer(ovis_path, max_length=256)
    ovis_token = ovis_tokenizer.encode(text)
    import pdb
    pdb.set_trace()
