import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_module(config):
    dataset_name = config.pop("dataset")
    if dataset_name == "mnist":
        return MNISTSequenceDataModule(**config)
    raise ValueError(f"Dataset {dataset_name} not supported")


class MNISTSequenceDataModule(pl.LightningDataModule):

    def __init__(self, bsz=128, num_workers=4, input_type="raw", data_dir="./data"):
        super().__init__()
        self.bsz = bsz
        self.num_workers = num_workers
        self.data_dir = data_dir

        # Constants for your task
        self.SEQ_LENGTH = 784  # 28x28 pixels flattened
        self.IN_DIM = 1  # grayscale

        # Define transforms once
        transform_list = [transforms.ToTensor()]
        if input_type == "tokenized":
            transform_list.append(
                transforms.Lambda(lambda x: (x.view(self.IN_DIM, self.SEQ_LENGTH).t() * 255).int()),
            )
        elif input_type == "raw":
            transform_list.append(
                transforms.Lambda(lambda x: x.view(self.IN_DIM, self.SEQ_LENGTH, -1).float())
            )
        else:
            raise ValueError(f"Input type {input_type} not supported")
        self.transform = transforms.Compose(transform_list)
        self.supported_tasks = ["classification", "generation"]

    def prepare_data(self):
        # download once
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def token2raw(cls, x):
        return x.unsqueeze(-1).float() / 255.0

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
