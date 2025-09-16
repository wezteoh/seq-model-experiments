"""
Adapted from https://github.com/state-spaces/s4
"""

from collections import namedtuple
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from src.modules.s4_block import (
    S4Block,  # Can use full version instead of minimal S4D standalone below
)


class S4Model(nn.Module):

    def __init__(
        self,
        d_input=None,
        vocab_size=None,
        output_value_size=None,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        d_state=64,
        prenorm=False,
        lr=0.001,
        wd=0.0,
        input_type="token",
        output_type="logits",
        pooling=False,
        backend="cuda",
    ):
        super().__init__()

        self.prenorm = prenorm
        self.pooling = pooling
        self.output_type = output_type

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        if input_type == "token":
            self.encoder = nn.Embedding(vocab_size, d_model)
        elif input_type == "raw":
            self.encoder = nn.Linear(d_input, d_model)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4Block(
                    d_model,
                    dropout=dropout,
                    transposed=True,
                    lr=lr,
                    wd=wd,
                    d_state=d_state,
                    backend=backend,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        if output_type == "logits":
            self.decoder = nn.Linear(d_model, vocab_size)
        elif output_type == "values":
            self.decoder = nn.Linear(d_model, output_value_size)

    def default_states(self, *batch_shape, device=None):
        return [layer.default_state(*batch_shape, device=device) for layer in self.s4_layers]

    def forward(self, x, states=[None] * 4):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        next_states = []
        for layer, norm, dropout, state in zip(self.s4_layers, self.norms, self.dropouts, states):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, next_state = layer(z, state=state)
            next_states.append(next_state)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        if self.pooling:
            x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, ..., d_model) -> (B, ..., d_output)

        causal_lm_output = namedtuple("CausalLMOutput", [self.output_type, "hidden_states"])
        return causal_lm_output(**{self.output_type: x}, hidden_states=next_states)

    def _setup_step(self, **kwargs):
        for layer in self.s4_layers:
            layer.setup_step(**kwargs)

    def step(self, x, states):
        x = self.encoder(x)  # (B, d_model)

        next_states = []
        for layer, norm, dropout, state in zip(self.s4_layers, self.norms, self.dropouts, states):
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, next_state = layer.step(z, state)
            next_states.append(next_state)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

        x = self.decoder(x)

        return x, next_states

    def generate(self, x, max_length=785, **kwargs):
        self._setup_step()
        if self.output_type == "logits":
            return self.token_generate(x, max_length, **kwargs)
        elif self.output_type == "values":
            return self.raw_generate(x, max_length, **kwargs)

    def raw_generate(self, x, max_length=785, **kwargs):
        raise NotImplementedError("Raw generation not implemented for S4")

    def token_generate(
        self,
        input_ids: LongTensor,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
        output2input_preprocess_fn: Callable[[Tensor], Tensor] = None,
    ) -> LongTensor:

        max_new_length = max_length - input_ids.shape[1]
        output = input_ids

        prefix, head = input_ids[:, :-1], input_ids[:, -1:]

        if output2input_preprocess_fn is not None:
            prefix = output2input_preprocess_fn(prefix)
            head = output2input_preprocess_fn(head).squeeze(dim=1)

        init_states = self.default_states(prefix.shape[0], device=prefix.device)
        next_states = self(prefix, init_states).hidden_states

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                logits, next_states = self.step(head, next_states)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][..., -1:]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                return output
            output = torch.cat([output, next_token], dim=1)
            head = next_token
            if output2input_preprocess_fn is not None:
                head = output2input_preprocess_fn(head).squeeze(dim=1)
        return output
