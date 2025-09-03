import pytorch_lightning as pl
import torch
import wandb

from src.utils import bw_to_rgb, unflatten_images


class ImagePrefixSamplerCallback(pl.Callback):
    def __init__(self, num_samples=16, every_n_epochs=1, sample_prefix_length=10):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.sample_prefix_length = sample_prefix_length

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or trainer.state.fn != "fit":
            return  # only run every N epochs

        val_loader = trainer.datamodule.val_dataloader()
        x, *_ = next(iter(val_loader))
        x = x[: self.num_samples]
        examples = x.clone().numpy()

        examples = unflatten_images(examples, shape=(28, 28))
        examples = bw_to_rgb(examples)
        examples = [wandb.Image(example) for example in examples]
        wandb.log({"examples": examples}, commit=False)

        x = x[: self.num_samples, : self.sample_prefix_length].to(self.device)
        x = torch.cat([torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device), x], dim=1)
        samples = self.model.prefix_sample(x)
        samples = samples[:, 1:].cpu().numpy()
        samples = unflatten_images(samples, shape=(28, 28))
        samples = bw_to_rgb(samples)
        samples = [wandb.Image(sample) for sample in samples]
        wandb.log({"samples": samples}, commit=False)

        if pl_module.training:  # restore training mode
            pl_module.train()
