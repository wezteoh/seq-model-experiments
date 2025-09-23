import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn import RMSNorm  # only after torch 2.4
except ImportError:
    from .basic import RMSNorm


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w3(x) * (self.activation(self.w1(x)))
        x = self.w2(x)
        return x


class RoPE(nn.Module):
    def __init__(
        self, theta: float = 10000.0, d_k: int = 1024, max_seq_len: int = 1024, device: str = None
    ):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.idx = torch.arange(0, max_seq_len)
        self.theta = theta ** -(torch.arange(0, d_k, 2) / d_k)
        self.device = device
        self.build_cache()

    def build_cache(self):
        idx_theta = einops.einsum(self.idx, self.theta, "i, j -> i j")
        cos_cached = torch.cos(idx_theta).to(self.device)
        sin_cached = torch.sin(idx_theta).to(self.device)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        if token_positions is None:
            cos_cached = self.cos_cached[: x.shape[-2]]
            sin_cached = self.sin_cached[: x.shape[-2]]
        else:
            cos_cached = self.cos_cached[token_positions].unsqueeze(-3)  # add head dimension
            sin_cached = self.sin_cached[token_positions].unsqueeze(-3)  # add head dimension

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x1 = x_even * cos_cached - x_odd * sin_cached
        x2 = x_odd * cos_cached + x_even * sin_cached

        x = torch.stack([x1, x2], dim=-1).flatten(start_dim=-2, end_dim=-1).to(x.dtype)
        return x


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        max_seq_len: int = 1024,
        device: str = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.rope = (
            RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
            if use_rope
            else None
        )

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        inference_params=None,
        layer_idx=None,
    ) -> torch.Tensor:

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, "b seq_len (h d_k) -> b h seq_len d_k", h=self.num_heads)
        k = einops.rearrange(k, "b seq_len (h d_k) -> b h seq_len d_k", h=self.num_heads)
        v = einops.rearrange(v, "b seq_len (h d_v) -> b h seq_len d_v", h=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if inference_params is not None:
            prev_keys = inference_params.key_value_memory_dict.get(f"attn.{layer_idx}.keys", None)
            prev_values = inference_params.key_value_memory_dict.get(
                f"attn.{layer_idx}.values", None
            )
            if prev_keys is not None:
                k = torch.cat([prev_keys, k], dim=-2)
            if prev_values is not None:
                v = torch.cat([prev_values, v], dim=-2)
            inference_params.key_value_memory_dict[f"attn.{layer_idx}.keys"] = k
            inference_params.key_value_memory_dict[f"attn.{layer_idx}.values"] = v
            x = F.scaled_dot_product_attention(q, k, v, None)  # b, h, seq_len, d_v

        else:
            mask = torch.tril(torch.ones(x.shape[1], x.shape[1]))
            mask = mask.bool().to(x.device)

            x = F.scaled_dot_product_attention(q, k, v, mask)  # b, h, seq_len, d_v

        x = einops.rearrange(x, "b h seq_len d_v -> b seq_len (h d_v)")

        x = self.output_proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        max_seq_len: int = 1024,
        device: str = None,
    ):
        super().__init__()
        self.attn = CausalMultiheadSelfAttention(
            d_model, num_heads, use_rope, theta, max_seq_len, device
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        inference_params=None,
        layer_idx=None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x), token_positions, inference_params=inference_params, layer_idx=layer_idx
        )
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    q = torch.ones(1, 1, 3).float()
    k = torch.tensor([0, 1, 0]).unsqueeze(-1).unsqueeze(0).float()
    k = k.expand(1, 3, 3)
    v = torch.tensor([0, 100, 200]).unsqueeze(-1).unsqueeze(0).float()
    v = v.expand(1, 3, 1)

    out = F.scaled_dot_product_attention(q, k, v, None)
    print(out)
