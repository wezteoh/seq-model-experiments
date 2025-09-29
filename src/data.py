import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from src.data_utils import Tokenizer, Vocab, generate_induction_head
from src.utils import cast_floats_by_trainer_precision


def get_output2input_preprocess_fn(preprocess_fn_name):
    if preprocess_fn_name is None:
        return None
    if preprocess_fn_name == "mnist_token2raw":
        return MNISTSequenceGenerationDataModule.token2raw
    raise ValueError(f"Preprocess function {preprocess_fn_name} not supported")


def get_data_module_class(dataset_name):
    if dataset_name == "mnist_generation":
        return MNISTSequenceGenerationDataModule
    if dataset_name in ("icl"):
        return ICLDataModule
    if dataset_name == "trajectory_generation":
        return TrajectoryDataModule
    raise ValueError(f"Dataset {dataset_name} not supported")


class MNISTSequenceGenerationDataModule(pl.LightningDataModule):

    # class variables, to allow model to access outside of training
    SUPPORTED_TASK_TYPE = "generation"

    @classmethod
    def token2raw(cls, x):
        return x.unsqueeze(-1).float() / 255.0

    @classmethod
    def pad_start_of_sequence(cls, x):
        return torch.cat([torch.zeros_like(x[:, 0:1]), x], dim=1)

    def __init__(
        self,
        bsz=128,
        num_workers=4,
        input_type="raw",
        target_type="token",
        data_dir="./data",
    ):
        super().__init__()
        self.bsz = bsz
        self.num_workers = num_workers
        self.data_dir = os.path.expanduser(data_dir)
        self.input_type = input_type
        self.target_type = target_type
        self.seq_length = 784
        # Define transforms once
        transform_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.view(self.seq_length).t() * 255).int()),
        ]
        self.transform = transforms.Compose(transform_list)

    def prepare_data(self):
        # download once
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "validate"):
            self.val_ds = torchvision.datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        if stage in (None, "fit"):
            self.train_ds = torchvision.datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
        if stage in (None, "test"):
            self.test_ds = torchvision.datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):

        x, _ = batch
        y = x.clone()
        # slide x one step back
        x = self.__class__.pad_start_of_sequence(x)[:, :-1]

        if self.input_type == "raw":
            x = self.__class__.token2raw(x)
        if self.target_type == "raw":
            y = self.__class__.token2raw(y)

        x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)
        y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)
        return x, y

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.bsz,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.bsz,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.bsz,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


"""Synthetic datasets to test in-context learning ability."""


class ICLDataModule(pl.LightningDataModule):
    _name_ = "icl"
    SUPPORTED_TASK_TYPE = "generation"

    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        vocab_size: int,
        input_seq_len: int,
        number_duplicates_per_epoch: int = 0,
        seed: int = 42,
        batch_size: int = 32,
        split_train_test: bool = False,
        induction_len: int = 1,
        induction_num_triggers: int = 1,
        allow_dot: bool = False,
        max_copy_len: int = 10,
        test_seq_len: int = None,
        num_keys: int = 1,  # number of keys for associative recall,
        data_dir: str = None,
        # Align interface with MNIST module
        bsz: int = None,
        num_workers: int = 4,
        input_type: Literal["token"] = "token",
        target_type: Literal["token"] = "token",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.input_seq_len = input_seq_len
        self.vocab_size = vocab_size
        self.copy_method = "induction_head"
        self.number_duplicates_per_epoch = number_duplicates_per_epoch
        self.seed = seed
        self.batch_size = batch_size
        # Map MNIST-like args
        self.bsz = bsz if bsz is not None else batch_size
        self.num_workers = num_workers
        self.input_type = input_type
        self.target_type = target_type
        self.split_train_test = split_train_test  # let the same copy chars appear in train/test
        self.induction_len = induction_len
        self.induction_num_triggers = induction_num_triggers
        self.allow_dot = allow_dot
        self.max_copy_len = max_copy_len
        self.data_dir = os.path.expanduser(data_dir)

        if test_seq_len is not None:
            self.test_seq_len = test_seq_len
        else:
            self.test_seq_len = input_seq_len
        self.num_keys = num_keys

        special_vocabs = {"copy_prefix": "=>", "noop": "."}
        self.special_vocabs = special_vocabs
        self.vocab = Vocab(vocab_size - len(special_vocabs), special_vocabs=special_vocabs)
        self.tokenizer = Tokenizer(self.vocab)

        self.num_extra_seq_len = 2

        if self.copy_method == "induction_head":
            self.copy_f = self.generate_induction_head
            self.num_extra_seq_len = 1 + self.induction_len
        else:
            self.copy_f = None

        if self.number_duplicates_per_epoch > 0:
            self.duplicate_ex = self.generate_example()
            self.duplicate_index = max(int(self.num_examples / self.number_duplicates_per_epoch), 1)
        else:
            self.duplicate_ex = None
            self.duplicate_index = -1

        self.total_seq_len = self.input_seq_len + self.num_extra_seq_len

    def prepare_data(self):
        # Nothing to download; datasets are generated in setup
        pass

    def generate_induction_head(self, seqlen=None, valid_chars=None):
        return generate_induction_head(
            self.vocab,
            seqlen if seqlen is not None else self.input_seq_len,
            self.special_vocabs["copy_prefix"],
            self.induction_len,
            self.induction_num_triggers,
            self.rng,
            valid_chars=valid_chars,
        )

    def generate_example(self, seqlen=None, valid_chars=None):
        vocab_seq = self.copy_f(seqlen=seqlen, valid_chars=valid_chars)
        return self.tokenizer.tokenize(vocab_seq, return_tensor=True)

    def setup(self, stage=None):
        train_tensor = test_tensor = None
        if self.data_dir is not None:
            try:
                train_tensor = torch.load(
                    os.path.join(
                        self.data_dir,
                        f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    )
                )
                test_tensor = torch.load(
                    os.path.join(
                        self.data_dir,
                        f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    )
                )
            except Exception:
                pass

        if train_tensor is None or test_tensor is None:
            if hasattr(self, "dataset"):
                return
            self.rng = np.random.default_rng(self.seed)

            if self.split_train_test:
                all_vocab = self.vocab.non_special_vocab
                train_vocab = set(
                    self.rng.choice(all_vocab, size=len(all_vocab) // 2, replace=False)
                )
                test_vocab = set(all_vocab) - train_vocab
                train_vocab = list(train_vocab)
                test_vocab = list(test_vocab)
            else:
                train_vocab = None
                test_vocab = None

            all_examples = []
            for i, (example_count, valid_vocab) in enumerate(
                zip([self.num_examples, self.num_test_examples], [train_vocab, test_vocab])
            ):
                examples = torch.stack(
                    [
                        self.generate_example(
                            seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
                            valid_chars=valid_vocab,
                        )["input_ids"]
                        for _ in tqdm(range(example_count))
                    ]
                )
                examples = torch.unique(examples, dim=0, sorted=False).tolist()

                while len(examples) < example_count:
                    new_example = self.generate_example(
                        seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
                        valid_chars=valid_vocab,
                    )["input_ids"].tolist()
                    if new_example not in examples:
                        examples.append(new_example)

                self.rng.shuffle(examples)
                all_examples.append(torch.LongTensor(examples))

            # all_examples = torch.concat(all_examples)
            train_tensor = torch.stack(
                [torch.stack([example[:-1], example[1:]]) for example in all_examples[0]]
            )
            test_tensor = torch.stack(
                [torch.stack([example[:-1], example[1:]]) for example in all_examples[1]]
            )
            test_tensor[:, 1, : -1 * (self.num_extra_seq_len - 1)] = -100
            if self.copy_method in ["assoc_recall"]:
                test_tensor[:, 1, :-1] = -100
            if self.copy_method in ["majority", "fom1"]:
                train_tensor[:, 1, : -1 * (self.num_extra_seq_len - 1)] = -100

            if self.data_dir is not None:
                torch.save(
                    train_tensor,
                    os.path.join(
                        self.data_dir,
                        f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    ),
                )
                torch.save(
                    test_tensor,
                    os.path.join(
                        self.data_dir,
                        f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    ),
                )

        self.dataset = {
            "train": TensorDataset(train_tensor[:, 0, :], train_tensor[:, 1, :]),
            "test": TensorDataset(test_tensor[:, 0, :], test_tensor[:, 1, :]),
        }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["train"], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.bsz,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class TrajectoryDataModule(pl.LightningDataModule):

    SUPPORTED_TASK_TYPE = "generation"

    @classmethod
    def flatten_entity_dim(cls, x):
        return x.flatten(start_dim=-2)

    @classmethod
    def unflatten_entity_dim(cls, x):
        return x.reshape((*x.shape[:-1], -1, 2))

    @classmethod
    def normalize(cls, x, width, height):
        x = x.clone()
        x[..., 0] = 2 * x[..., 0] / float(width) - 1
        x[..., 1] = 2 * x[..., 1] / float(height) - 1
        return x

    @classmethod
    def unnormalize(cls, x, width, height):
        x = x.clone()
        x[..., 0] = (x[..., 0] + 1) * float(width) / 2
        x[..., 1] = (x[..., 1] + 1) * float(height) / 2
        return x

    @classmethod
    def pad_start_of_sequence(cls, x):
        return torch.cat([torch.ones_like(x[:, 0:1]), x], dim=1) * -1.0

    def __init__(
        self,
        data_dir: str = None,
        bsz: int = 32,
        num_workers: int = 4,
        traj_space_width: int = 94,
        traj_space_height: int = 50,
        pad_start_of_sequence: bool = True,
        input_type: Literal["raw"] = "raw",
        target_type: Literal["raw"] = "raw",
    ):
        super().__init__()
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.bsz = bsz
        self.num_workers = num_workers
        self.traj_space_width = traj_space_width
        self.traj_space_height = traj_space_height
        self.pad_start_of_sequence = pad_start_of_sequence

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data = np.load(self.data_dir / "train_clean.p", allow_pickle=True)
        test_data = np.load(self.data_dir / "test_clean.p", allow_pickle=True)
        self.dataset = {
            "train": train_data,
            "test": test_data,
        }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["train"], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.bsz,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):

        x = batch
        x = self.__class__.normalize(x, self.traj_space_width, self.traj_space_height)
        x = self.__class__.flatten_entity_dim(x)
        y = x.clone()

        # slide x one step back
        if self.pad_start_of_sequence:
            x = self.__class__.pad_start_of_sequence(x)
        else:
            x = x[:, :-1]
            y = y[:, 1:]

        x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)
        y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)
        return x, y
