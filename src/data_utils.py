from typing import Dict

import numpy as np
import torch


class Vocab:
    """Custom vocab."""

    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold copy_prefix and noop/pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs
        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(vocab)

    def get_next_vocab(self, token: str):
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def copy_prefix(self):
        return self.special_vocabs["copy_prefix"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    @property
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


class Tokenizer:
    """Custom Tokenizer for our own vocab."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def tokenize(self, text: str, return_tensor=False, mask_input=False):
        input_ids = [self.vocab.get_id(t) for t in text.split()]
        if self.vocab.get_id(self.vocab.copy_prefix) not in input_ids:
            raise ValueError("Input text must contain copy_prefix token.")
        copy_prefix_pos = input_ids.index(self.vocab.get_id(self.vocab.copy_prefix))
        labels = input_ids
        if mask_input:
            # Mask the input tokens for loss but do not mask the copied token
            labels = [-100] * (copy_prefix_pos + 1) + labels[copy_prefix_pos + 1 :]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])


def generate_start_seq(vocab: Vocab, input_seq_len: int, rng: np.random.Generator):
    """Generate token sequence up to and including the copy_prefix token."""
    vocab_seq = rng.choice(
        vocab.vocab,
        input_seq_len,
        replace=True,
        # Do not generate any special tokens
        p=[
            1 / (len(vocab) - len(vocab.special_tokens)) if p not in vocab.special_tokens else 0
            for p in vocab.vocab
        ],
    )
    vocab_seq = np.append(vocab_seq, vocab.copy_prefix)
    return vocab_seq.tolist()


def generate_induction_head(
    vocab: Vocab,
    input_seq_len: int,
    copy_prefix: str,
    induction_len: int,
    num_triggers: int,
    rng: np.random.Generator,
    valid_chars: list = None,
):
    """Generate sequence where the copy prefix is inserted into the input
    and then the character after the copy prefix is copied at the end.
    """
    if valid_chars is not None:
        raise NotImplementedError("Valid chars not implemented for induction heads.")
    vocab_seq = generate_start_seq(vocab, input_seq_len, rng)
    if rng.uniform() < 0.5:
        num_triggers = 1
    pos = sorted(rng.integers(input_seq_len - (1 + induction_len), size=num_triggers))
    pos_filtered = []
    for i, p in enumerate(pos):
        if i == 0:
            pos_filtered.append(p)
        elif p - pos_filtered[-1] > induction_len:
            pos_filtered.append(p)
    to_copy = [vocab_seq[pos_filtered[0] + 1 + i] for i in range(induction_len)]
    for pos in pos_filtered:
        vocab_seq[pos] = copy_prefix
        for i in range(induction_len):
            vocab_seq[pos + 1 + i] = to_copy[i]
    # if valid_chars is not None and to_copy not in valid_chars:
    #     vocab_seq[pos+1] = rng.choice(valid_chars)
    #     to_copy = vocab_seq[pos+1]
    vocab_seq = vocab_seq + to_copy
    return " ".join(vocab_seq)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    x = x.clone()
    x = (x - mean) / std
    return x


def unnormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    x = x.clone()
    x = x * std
    x = x + mean
    return x


def flatten_trajectory_entity_dim(x: torch.Tensor):
    return x.flatten(start_dim=-2)


def unflatten_trajectory_entity_dim(x: torch.Tensor):
    return x.reshape((*x.shape[:-1], -1, 2))
