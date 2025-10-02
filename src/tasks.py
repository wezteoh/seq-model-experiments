import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.data_utils import (
    discretize_trajectory,
    flatten_trajectory_entity_dim,
    normalize,
    reverse_discretize_trajectory,
    unflatten_trajectory_entity_dim,
    unnormalize,
)
from src.models import get_model
from src.utils import cast_floats_by_trainer_precision


class BaseSequenceTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config, logger=True)

        self.model = get_model(
            name=self.hparams.model.name,
            args=self.hparams.model.args,
            device="cuda" if config.trainer.accelerator == "gpu" else "cpu",
        )

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        raise NotImplementedError("Subclass must implement this method")

    def model_output2input_preprocess(self, x: torch.tensor, prefix: torch.tensor):
        raise NotImplementedError("Subclass must implement this method")

    def loss(self, pred, y, **w):
        if self.hparams.task.loss == "cross_entropy":
            output = pred.logits
            output = output.reshape(-1, output.shape[-1])
            y = y.reshape(-1).long()
            loss = F.cross_entropy(output, y, ignore_index=-100)
        elif self.hparams.task.loss == "mse":
            output = pred.values
            if self.hparams.trainer.accelerator == "cpu":
                loss = F.mse_loss(output.float(), y.float())
            else:
                loss = F.mse_loss(output, y)
        return loss

    def forward(self, x):
        """Passes a batch through the encoder, backbone, and decoder"""

        if self.hparams.model.args.input_type == "value":
            assert len(x.shape) == 3  # B,T,C
        elif self.hparams.model.args.input_type == "token":
            assert len(x.shape) == 2  # B,T

        pred = self.model.forward(x)

        return pred

    def training_step(self, batch, batch_idx):
        x, y = self.make_model_inputs_and_targets(batch)
        pred = self.forward(x)
        loss = self.loss(pred, y)

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        record_step = {}
        for metric in self.hparams.metrics:
            if metric == "loss":
                record_step[f"trainer_{metric}"] = loss.item()

        self.log_dict(
            record_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            # sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = self.make_model_inputs_and_targets(batch)
        pred = self.forward(x)
        loss = self.loss(pred, y)

        record = {}
        for metric, metric_detail in self.hparams.metrics.items():
            if metric == "loss":
                record[f"validation_{metric}"] = loss.item()
            elif self.hparams.task.target_type == "token" and metric == "accuracy":
                pred_class = pred.logits.argmax(dim=-1)
                record[f"validation_{metric}"] = (pred_class == y).float().mean().item()

        self.log_dict(
            record,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            # sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, pred = self.forward(batch)
        loss = self.loss(pred, y)
        record = {"test/loss": loss}

        if self.hparams["task_type"] == "classification":
            pred_class = pred.logits.argmax(dim=-1)
            record["test_acc"] = (pred_class == y).float().mean()

        self.log_dict(
            record,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            # sync_dist=True,
        )

        return loss

    def configure_optimizers(
        self,
    ):
        lr = self.hparams.optimizer.lr
        weight_decay = self.hparams.optimizer.weight_decay

        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(self.hparams.get("beta1", 0.9), self.hparams.get("beta2", 0.95)),
        )

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )
        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        if self.hparams.optimizer.lr_schedule:
            total_steps = self.hparams.optimizer.lr_schedule.total_steps or getattr(
                self.trainer, "estimated_stepping_batches", None
            )
            if total_steps is None:
                raise ValueError(
                    "total_steps not set. Pass total_steps=... to the module "
                    "or let Lightning set trainer.estimated_stepping_batches by calling trainer.fit first."
                )
            max_lrs = [g.get("lr", lr) for g in optimizer.param_groups]
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.hparams.optimizer.lr_schedule.pct_start,
                anneal_strategy="cos",
                cycle_momentum=False,
                div_factor=self.hparams.optimizer.lr_schedule.div_factor,
                final_div_factor=self.hparams.optimizer.lr_schedule.final_div_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycleLR updates every step
                    "frequency": 1,
                },
            }
        else:
            return optimizer


class MNISTGenerationTask(BaseSequenceTask):

    def token2value(self, x: torch.tensor):
        return x.unsqueeze(-1).float() / 255.0

    def pad_start_of_sequence(self, x: torch.tensor):
        return torch.cat([torch.zeros_like(x[:, 0:1]), x], dim=1)

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        x = batch
        y = x.clone()
        # slide x one step back
        x = self.pad_start_of_sequence(x)[:, :-1]

        if self.hparams.task.input_type == "value":
            x = self.token2value(x)
        if self.hparams.task.target_type == "value":
            y = self.token2value(y)

        x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)
        y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)
        return x, y

    def model_output2input_preprocess(self, x: torch.tensor, **kwargs):
        if self.hparams.task.input_type == "value":
            x = self.token2value(x)
            x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)
        return x


class TrajectoryGenerationTask(BaseSequenceTask):
    def __init__(self, config):
        super().__init__(config)
        if self.hparams.task.input_type == "value" or self.hparams.task.target_type == "value":
            self.data_mean = torch.tensor(self.hparams.task.data_mean)
            self.data_std = torch.tensor(self.hparams.task.data_std)
            self.diff_as_target = self.hparams.task.diff_as_target
            if self.hparams.task.diff_as_target:
                assert (
                    self.hparams.task.input_type == "value"
                    and self.hparams.task.target_type == "value"
                )
                self.diff_mean = torch.tensor(self.hparams.task.diff_mean)
                self.diff_std = torch.tensor(self.hparams.task.diff_std)
        if self.hparams.task.target_type == "token" or self.hparams.task.input_type == "token":
            self.space_width_partition_gap = (
                self.hparams.task.space_width / self.hparams.task.space_width_partition_count
            )
            self.space_height_partition_gap = (
                self.hparams.task.space_height / self.hparams.task.space_height_partition_count
            )
            self.space_width_partition_bias = self.space_width_partition_gap / 2
            self.space_height_partition_bias = self.space_height_partition_gap / 2

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        if self.hparams.task.input_type == "token" or self.hparams.task.target_type == "token":
            discretized_batch, discretized_batch_id = discretize_trajectory(
                batch,
                self.space_width_partition_gap,
                self.space_height_partition_gap,
                self.space_width_partition_bias,
                self.space_height_partition_bias,
                self.hparams.task.space_width_partition_count,
                self.hparams.task.space_height_partition_count,
            )
        if self.hparams.task.input_type == "value":
            batch_n = normalize(
                discretized_batch if self.hparams.task.target_type == "token" else batch,
                self.data_mean,
                self.data_std,
            )
            batch_nf = flatten_trajectory_entity_dim(batch_n)
            x = batch_nf[:, :-1]
            x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)
        else:
            x = discretized_batch_id[:, :-1]

        if self.hparams.task.target_type == "token":
            y = discretized_batch_id[:, 1:]
        elif self.diff_as_target:
            diff = torch.diff(batch, dim=1)
            diff_n = normalize(diff, self.diff_mean, self.diff_std)
            diff_nf = flatten_trajectory_entity_dim(diff_n)
            y = diff_nf
            y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)
        else:
            y = batch_nf[:, 1:]
            y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)

        return x, y

    def model_output2input_preprocess(self, x: torch.tensor, prefix: torch.tensor):
        if self.diff_as_target:
            diff = unflatten_trajectory_entity_dim(x)
            diff_un = unnormalize(diff, self.diff_mean, self.diff_std)
            value_un = unflatten_trajectory_entity_dim(prefix[:, -1:]) + diff_un
            value_n = normalize(value_un, self.data_mean, self.data_std)
            x = flatten_trajectory_entity_dim(value_n)
            x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)

        elif self.hparams.task.target_type == "token" and self.hparams.task.input_type == "value":
            x = reverse_discretize_trajectory(
                x,
                self.space_width_partition_gap,
                self.space_height_partition_gap,
                self.space_width_partition_bias,
                self.space_height_partition_bias,
                self.hparams.task.space_width_partition_count,
                self.hparams.task.space_height_partition_count,
            )
            x = normalize(x, self.data_mean, self.data_std)
            x = flatten_trajectory_entity_dim(x)
            x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)
        return x

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = super().validation_step(batch, batch_idx, dataloader_idx)
        record = {}
        for metric, metric_detail in self.hparams.metrics.items():
            if metric == "ade":
                batch_prefix = batch[:, : metric_detail["prefix_length"]]
                x, _ = self.make_model_inputs_and_targets(batch)
                x = x[:, : metric_detail["prefix_length"]]
                assert metric_detail["prefix_length"] > 0
                output = self.model.generate(
                    x,
                    max_length=batch.shape[1],
                    output2input_preprocess_fn=self.model_output2input_preprocess,
                )
                if self.hparams.task.target_type == "token":
                    pred = reverse_discretize_trajectory(
                        output,
                        self.space_width_partition_gap,
                        self.space_height_partition_gap,
                        self.space_width_partition_bias,
                        self.space_height_partition_bias,
                        self.hparams.task.space_width_partition_count,
                        self.hparams.task.space_height_partition_count,
                    )
                elif self.hparams.task.diff_as_target:
                    diff = unflatten_trajectory_entity_dim(output)
                    diff_un = unnormalize(diff, self.diff_mean, self.diff_std)
                    pred = batch_prefix[:, -1:] + diff_un
                else:
                    pred = unflatten_trajectory_entity_dim(output)
                    pred = unnormalize(pred, self.data_mean, self.data_std)

                displacement = torch.norm(pred - batch[:, metric_detail["prefix_length"] :], dim=-1)
                record[f"validation_{metric}"] = displacement.mean().item()
                if "grouping" in metric_detail:
                    for group_name, idxs in metric_detail["grouping"].items():
                        record[f"validation_{metric}_{group_name}"] = (
                            displacement[..., idxs].mean().item()
                        )

        self.log_dict(
            record,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )
        return loss
