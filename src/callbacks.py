import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb

from src.data_utils import (
    reverse_discretize_trajectory,
    unflatten_trajectory_entity_dim,
    unnormalize,
)
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
        batch = batch[0][: self.num_samples]
        examples = batch.clone().cpu().numpy()

        examples = unflatten_images(examples, shape=(28, 28))
        examples = bw_to_rgb(examples)
        examples = [wandb.Image(example) for example in examples]
        wandb.log({"examples": examples}, commit=False)

        # pad dummy start of sequence
        batch = batch.to(pl_module.device)
        sample_prefix = batch[:, : self.sample_prefix_length]
        x, _ = pl_module.make_model_inputs_and_targets(batch)
        x_prefix = x[:, : self.sample_prefix_length + 1]
        pred = pl_module.model.generate(
            x_prefix,
            max_length=self.max_length + 1,
            top_k=self.top_k,
            eos_token_id=-1,
            output2input_preprocess_fn=pl_module.model_output2input_preprocess,
        )
        if pl_module.hparams.task.target_type == "value":
            pred = pred.int8().squeeze(-1)

        samples = torch.cat([sample_prefix, pred], dim=1).cpu().numpy()
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
        batch = batch[: self.num_samples]

        if epoch == 0:
            examples = batch.clone().cpu().numpy()
            example_videos = []
            for i, example in enumerate(examples):
                example = example[:: self.downsampling_ratio]
                frames = create_frames_from_trajectory(example, game=self.game)
                video_path = f"{self.sample_dir}/example_{i}.mp4"
                create_video_from_frames(frames, video_path, fps=self.fps)
                example_videos.append(wandb.Video(video_path, format="mp4"))
            wandb.log({"examples": example_videos}, commit=False)

        batch = batch.to(pl_module.device)
        sample_prefix = batch[:, : self.sample_prefix_length]
        x, _ = pl_module.make_model_inputs_and_targets(batch)
        x_prefix = x[:, : self.sample_prefix_length]
        output = pl_module.model.generate(
            x_prefix,
            max_length=self.max_length,
            output2input_preprocess_fn=pl_module.model_output2input_preprocess,
        )
        if pl_module.hparams.task.target_type == "token":
            pred = reverse_discretize_trajectory(
                output,
                pl_module.space_width_partition_gap,
                pl_module.space_height_partition_gap,
                pl_module.space_width_partition_bias,
                pl_module.space_height_partition_bias,
                pl_module.hparams.task.space_width_partition_count,
                pl_module.hparams.task.space_height_partition_count,
            )
        elif pl_module.hparams.task.diff_as_target:
            diff = pl_module.maybe_unflatten_entity_dim(output)
            diff_un = unnormalize(diff, pl_module.diff_mean, pl_module.diff_std)
            diff_un_cumsum = torch.cumsum(diff_un, dim=1)
            pred = sample_prefix[:, -1:] + diff_un_cumsum
        else:
            pred = pl_module.maybe_unflatten_entity_dim(output)
            pred = unnormalize(pred, pl_module.data_mean, pl_module.data_std)

        samples = torch.cat([sample_prefix, pred], dim=1).cpu().numpy()
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
