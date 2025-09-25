from collections import namedtuple
from functools import partial

import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block

from src.inference import CustomGenerationMixin

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba2MixerModel(nn.Module, CustomGenerationMixin):
    def __init__(
        self,
        input_type: str,
        output_type: str,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int | None = None,
        out_value_size: int | None = None,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.input_type = input_type
        self.output_type = output_type

        # sanity check on args compatibility to input_type and output_type
        if self.input_type == "token" or self.output_type == "logits":
            assert vocab_size is not None
        if self.output_type == "values":
            assert out_value_size is not None

        if self.input_type == "token":
            self.encoder = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        else:
            self.encoder = nn.Linear(1, d_model, **factory_kwargs)
            self.encoder.bias._no_reinit = True
            # reinitalized as all 0s by initializer config from mamba_ssm package
            # could result in symmetry issue / exploding gradients on certain inputs

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        if self.output_type == "logits":
            self.head = nn.Linear(d_model, vocab_size, **factory_kwargs)
        elif self.output_type == "values":
            self.head = nn.Linear(d_model, out_value_size, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self, inputs, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs
    ):
        hidden_states = self.encoder(inputs)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        outputs = self.head(hidden_states)
        if self.output_type == "logits":
            CausalOutput = namedtuple("CausalOutput", ["logits"])
            return CausalOutput(logits=outputs)
        elif self.output_type == "values":
            CausalOutput = namedtuple("CausalOutput", ["values"])
            return CausalOutput(values=outputs)
