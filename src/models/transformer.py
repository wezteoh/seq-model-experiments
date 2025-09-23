import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference import CustomGenerationMixin
from src.modules.transformer_block import TransformerBlock

try:
    from torch.nn import RMSNorm  # only after torch 2.4
except ImportError:
    from src.modules.basic import RMSNorm

from collections import namedtuple


class TransformerModel(nn.Module, CustomGenerationMixin):

    def __init__(
        self,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        input_type: str = "token",
        output_type: str = "logits",
        vocab_size: int | None = None,
        out_value_size: int | None = None,
        device: str = None,
    ):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

        # sanity check on args compatibility to input_type and output_type
        if self.input_type == "token" or self.output_type == "logits":
            assert vocab_size is not None
        if self.output_type == "values":
            assert out_value_size is not None

        if self.input_type == "token":
            self.encoder = nn.Embedding(vocab_size, d_model)
        else:
            self.encoder = nn.Linear(1, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    use_rope,
                    theta,
                    max_seq_len=context_length,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor = None,
        inference_params=None,
        num_last_tokens=0,
    ) -> torch.Tensor:

        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)

        x = self.encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, position_ids, inference_params=inference_params, layer_idx=i)

        if num_last_tokens > 0:
            x = x[:, -num_last_tokens:]

        x = self.ln_final(x)
        x = self.lm_head(x)

        if self.output_type == "logits":
            CausalOutput = namedtuple("CausalOutput", ["logits"])
            return CausalOutput(logits=x)
        elif self.output_type == "values":
            CausalOutput = namedtuple("CausalOutput", ["values"])
            return CausalOutput(values=x)

        return x
