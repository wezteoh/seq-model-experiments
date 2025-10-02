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
        n_layer: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        input_type: str = "token",
        output_type: str = "logit",
        vocab_size: int | None = None,
        out_value_size: int | None = None,
        input_value_size: int | None = None,
        n_encoder_layer: int | None = None,
        split_count: int = 0,
        split_dim: int | None = None,
        device: str = None,
    ):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.split_count = split_count

        # sanity check on args compatibility to input_type and output_type
        if self.input_type == "token" or self.output_type == "logit":
            assert vocab_size is not None
        if self.output_type == "value":
            assert out_value_size is not None

        if self.input_type == "token":
            assert n_encoder_layer is None, "n_encoder_layer must be None for token input type"
            self.encoder = nn.Embedding(vocab_size, d_model)
        else:
            assert (
                n_encoder_layer is not None
            ), "n_encoder_layer must be provided for raw input type"
            assert (
                input_value_size is not None
            ), "input_value_size must be provided for raw input type"
            encoder_layers = [nn.Linear(input_value_size, d_model)]
            for _ in range(n_encoder_layer - 1):
                encoder_layers.append(
                    nn.Sequential(
                        nn.GELU(),
                        nn.Linear(d_model, d_model),
                    )
                )
            self.encoder = nn.Sequential(*encoder_layers)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_head,
                    d_ff,
                    use_rope,
                    theta,
                    max_seq_len=context_length,
                    device=device,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        if self.split_count > 1:
            self.split_head = nn.Sequential(
                nn.Linear(d_model, split_count * split_dim, bias=False),
                nn.GELU(),
            )
        self.lm_head = nn.Linear(
            d_model if self.split_count <= 1 else split_dim,
            vocab_size if self.output_type == "logit" else out_value_size,
            bias=False,
        )

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
        if self.split_count > 1:
            x = self.split_head(x)
            x = einops.rearrange(x, "... (n d) -> ... n d", n=self.split_count)
        x = self.lm_head(x)

        if self.output_type == "logit":
            CausalOutput = namedtuple("CausalOutput", ["logits"])
            return CausalOutput(logits=x)
        elif self.output_type == "value":
            CausalOutput = namedtuple("CausalOutput", ["values"])
            return CausalOutput(values=x)

        return x
