import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_output2input_preprocess_fn(preprocess_fn_name):
    if preprocess_fn_name is None:
        return None
    if preprocess_fn_name == "mnist_token2raw":
        return MNISTSequenceGenerationDataModule.token2raw
    raise ValueError(f"Preprocess function {preprocess_fn_name} not supported")


def get_data_module_class(dataset_name):
    if dataset_name == "mnist_generation":
        return MNISTSequenceGenerationDataModule
    raise ValueError(f"Dataset {dataset_name} not supported")


class MNISTSequenceGenerationDataModule(pl.LightningDataModule):

    # class variables, to allow model to access outside of training
    SUPPORTED_TASK_TYPE = "generation"
    SEQ_LENGTH = 784
    NUM_CLASSES = 256

    @classmethod
    def token2raw(cls, x):
        return x.unsqueeze(-1).float() / 255.0

    @classmethod
    def pad_start_of_sequence(cls, x):
        return torch.cat([torch.zeros_like(x[:, 0:1]), x], dim=1)

    def __init__(
        self, bsz=128, num_workers=4, input_type="raw", target_type="token", data_dir="./data"
    ):
        super().__init__()
        self.bsz = bsz
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.input_type = input_type
        self.target_type = target_type
        # Define transforms once
        transform_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.view(self.__class__.SEQ_LENGTH).t() * 255).int()),
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

    def sample_dataloader(self):
        return DataLoader(
            self.test_ds,
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
