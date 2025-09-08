import pytorch_lightning as pl
import torch
import wandb

from src.utils import bw_to_rgb, unflatten_images


class ImagePrefixSamplerCallback(pl.Callback):
    def __init__(self, num_samples=16, every_n_epochs=1, sample_prefix_length=10, max_length=785):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.sample_prefix_length = sample_prefix_length
        self.max_length = max_length

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or trainer.state.fn != "fit":
            return  # only run every N epochs

        sample_loader = trainer.val_dataloaders
        batch = next(iter(sample_loader))
        x = batch[0][: self.num_samples]
        examples = x.clone().cpu().numpy()

        examples = unflatten_images(examples, shape=(28, 28))
        examples = bw_to_rgb(examples)
        examples = [wandb.Image(example) for example in examples]
        wandb.log({"examples": examples}, commit=False)

        # pad dummy start of sequence
        x = trainer.datamodule.pad_start_of_sequence(x)
        x = x[:, : self.sample_prefix_length].to(pl_module.device)
        samples = pl_module.model.prefix_sample(
            x,
            output2input_preprocess_fn=pl_module.output2input_preprocess_fn,
            max_length=self.max_length,
        )
        # trim dummy start of sequence
        samples = samples[:, 1:].cpu().numpy()
        samples = unflatten_images(samples, shape=(28, 28))
        samples = bw_to_rgb(samples)
        samples = [wandb.Image(sample) for sample in samples]
        wandb.log({"samples": samples}, commit=False)

        if pl_module.training:  # restore training mode
            pl_module.train()


class InductionHeadTextSamplerCallback(pl.Callback):
    def __init__(self, num_samples=16, every_n_epochs=1, induction_length=1):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.induction_length = induction_length

    def on_validation_epoch_end(self, trainer, pl_module):
        at = wandb.Artifact(f"samples_epoch_{trainer.current_epoch}", type="samples")
        table = wandb.Table(columns=["epoch", "sample"])

        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or trainer.state.fn != "fit":
            return  # only run every N epochs

        sample_loader = trainer.val_dataloaders
        batch = next(iter(sample_loader))
        x = batch[0][: self.num_samples]

        samples = pl_module.model.generate(
            x, max_length=x.shape[1] + self.induction_length, top_k=1, eos_token_id=-1
        )
        samples = samples.cpu().numpy()
        samples = [trainer.datamodule.tokenizer.decode(sample) for sample in samples]
        for sample in samples:
            table.add_data(epoch, sample)
        at.add(table, "samples")
        wandb.log_artifact(at)

        if pl_module.training:  # restore training mode
            pl_module.train()
