from collections.abc import Callable
from typing import Optional

import torch
from mamba_ssm.utils.generation import (
    InferenceParams,
    modify_logit_for_repetition_penalty,
    sample,
    update_graph_cache,
)
from torch import Tensor
from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
    TextStreamer,
)


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    output_scores=False,
    streamer: Optional[TextStreamer] = None,
    output2input_preprocess_fn: Optional[Callable[[Tensor], Tensor]] = None,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        start.record()
    scores, sequences = [], [input_ids]
    sequences_cat = input_ids
    if output2input_preprocess_fn is not None:
        model_inputs = output2input_preprocess_fn(
            input_ids
        )  # handle token2raw transformation if needed
    else:
        model_inputs = input_ids
    while not should_stop(model_inputs, inference_params):
        logits = get_logits(model_inputs, inference_params)
        if output_scores:
            scores.append(logits.clone())
        inference_params.seqlen_offset += model_inputs.shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(logits, inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(logits, sequences_cat, repetition_penalty)
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        model_inputs = (
            output2input_preprocess_fn(sampled_tokens)
            if output2input_preprocess_fn is not None
            else sampled_tokens
        )
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class CustomGenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        if self.output_type == "logits":
            return self.token_generate(*args, **kwargs)
        elif self.output_type == "values":
            return self.raw_generate(*args, **kwargs)

    def raw_generate(self, *args, **kwargs):
        raise NotImplementedError

    def token_generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids,
            self,
            max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            output_scores=output_scores,
            **kwargs,
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences
