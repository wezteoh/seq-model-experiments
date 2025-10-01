import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb

from src.utils import (
    bw_to_rgb,
    create_frames_from_trajectory,
    create_video_from_frames,
    unflatten_images,
)


class ImagePrefixSamplerCallback(pl.Callback):
    def __init__(
        self,
        num_samples=16,
        every_n_epochs=1,
        sample_prefix_length=10,
        max_length=785,
        top_k=50,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.sample_prefix_length = sample_prefix_length
        self.max_length = max_length
        self.top_k = top_k

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
        samples = pl_module.model.generate(
            x,
            max_length=self.max_length,
            top_k=self.top_k,
            eos_token_id=-1,
            output2input_preprocess_fn=pl_module.output2input_preprocess_fn,
            precision=trainer.precision,
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


class TrajectoryPrefixSamplerCallback(pl.Callback):
    def __init__(
        self,
        num_samples=16,
        every_n_epochs=1,
        sample_dir="./samples",
        game="basketball",
        sample_prefix_length=10,
        max_length=100,
        downsampling_ratio=1,
        fps=2,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.sample_dir = os.path.expanduser(sample_dir)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=False)
        self.game = game
        self.sample_prefix_length = sample_prefix_length
        self.max_length = max_length
        self.downsampling_ratio = downsampling_ratio
        self.fps = fps

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or trainer.state.fn != "fit":
            return  # only run every N epochs

        sample_loader = trainer.val_dataloaders
        batch = next(iter(sample_loader))
        x = batch[: self.num_samples]

        if epoch == 0:
            examples = x.clone().cpu().numpy()
            example_videos = []
            for i, example in enumerate(examples):
                example = example[:: self.downsampling_ratio]
                frames = create_frames_from_trajectory(example, game=self.game)
                video_path = f"{self.sample_dir}/example_{i}.mp4"
                create_video_from_frames(frames, video_path, fps=self.fps)
                example_videos.append(wandb.Video(video_path, format="mp4"))
            wandb.log({"examples": example_videos}, commit=False)

        x = x[:, : self.sample_prefix_length].to(pl_module.device)
        samples = pl_module.model.generate(
            x,
            max_length=self.max_length,
            precision=trainer.precision,
            is_trajectory=True,
            output_diff=getattr(trainer.datamodule, "diff_as_target", False),
            data_mean=trainer.datamodule.data_mean,
            data_std=trainer.datamodule.data_std,
            diff_mean=(
                trainer.datamodule.diff_mean
                if getattr(trainer.datamodule, "diff_as_target", False)
                else None
            ),
            diff_std=(
                trainer.datamodule.diff_std
                if getattr(trainer.datamodule, "diff_as_target", False)
                else None
            ),
        )
        samples = samples.cpu().numpy()
        sample_videos = []
        for i, sample in enumerate(samples):
            sample = sample[:: self.downsampling_ratio]
            frames = create_frames_from_trajectory(sample, game=self.game)
            video_path = f"{self.sample_dir}/epoch_{epoch}_sample_{i}.mp4"
            create_video_from_frames(frames, video_path, fps=self.fps)
            sample_videos.append(wandb.Video(video_path, format="mp4"))
        wandb.log({"samples": sample_videos}, commit=False)

        if pl_module.training:  # restore training mode
            pl_module.train()
