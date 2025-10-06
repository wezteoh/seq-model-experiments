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
from typing import Literal


class TrajTransformerModel(nn.Module, CustomGenerationMixin):

    def __init__(
        self,
        context_length: int,
        n_entity: int,
        input_value_size: int,
        n_layer_spatial: int,
        d_entity: int,
        n_head_spatial: int,
        d_ff_spatial: int,
        n_layer_temporal: int,
        d_model: int,
        n_head_temporal: int,
        d_ff_temporal: int,
        out_value_size: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        input_type: Literal["value"] = "value",
        output_type: Literal["value"] = "value",
        device: str = None,
    ):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.n_entity = n_entity

        self.entity_encoder = nn.Sequential(
            nn.Linear(input_value_size, d_entity),
            nn.GELU(),
            nn.Linear(d_entity, d_entity),
        )

        self.spatial_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_entity,
                    n_head_spatial,
                    d_ff_spatial,
                    use_rope,
                    theta,
                    max_seq_len=context_length,
                    device=device,
                    causal=False,
                )
                for _ in range(n_layer_spatial)
            ]
        )
        self.spatial_ln = RMSNorm(d_entity)
        self.spatial2temporal = nn.Linear(d_entity * n_entity, d_model)
        self.temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_head_temporal,
                    d_ff_temporal,
                    use_rope,
                    theta,
                    max_seq_len=context_length,
                    device=device,
                    causal=True,
                )
                for _ in range(n_layer_temporal)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, out_value_size * n_entity, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor = None,
        inference_params=None,
        num_last_tokens=0,
    ) -> torch.Tensor:

        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)

        x = self.entity_encoder(x)
        for i, layer in enumerate(self.spatial_layers):
            x = layer(x)
        x = self.spatial_ln(x)
        x = einops.rearrange(x, "b t n d -> b t (n d)")
        x = self.spatial2temporal(x)
        for i, layer in enumerate(self.temporal_layers):
            x = layer(x, position_ids, inference_params=inference_params, layer_idx=i)

        if num_last_tokens > 0:
            x = x[:, -num_last_tokens:]

        x = self.ln_final(x)
        x = self.lm_head(x)
        x = einops.rearrange(x, "b t (n d) -> b t n d", n=self.n_entity)

        CausalOutput = namedtuple("CausalOutput", ["values"])
        return CausalOutput(values=x)
