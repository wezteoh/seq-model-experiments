import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import src.callbacks
from src.data import get_data_module_class, get_output2input_preprocess_fn


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config, logger=True)

        if torch.cuda.is_available() and self.hparams.use_fused_kernel:
            from src.model import CustomMixerModel

            # GPU path (src.model.CustomMixerModel signature: flat kwargs)
            self.model = CustomMixerModel(**self.hparams.model["args"]).to("cuda")
        else:
            # CPU path (src.torch_model.CustomMixerModel signature: args=Mamba2Config)
            from src.torch_model import CustomMixerModel, Mamba2Config

            args = Mamba2Config(**self.hparams.model["args"])
            self.model = CustomMixerModel(
                args=args, device="cuda" if config.trainer.accelerator == "gpu" else "cpu"
            )
        self.output2input_preprocess_fn = get_output2input_preprocess_fn(
            self.hparams.output2input_preprocess_fn_name
        )

    def loss(self, pred, y, **w):
        if self.hparams["loss"] == "cross_entropy":
            output = pred.logits
            output = output.reshape(-1, output.shape[-1])
            y = y.reshape(-1).long()
            loss = F.cross_entropy(output, y, ignore_index=-100)
        elif self.hparams["loss"] == "mse":
            output = pred.values
            output = output.reshape(-1, output.shape[-1])
            y = y.reshape(-1).long()
            loss = F.mse_loss(output, y)
        return loss

    def forward(self, batch):
        """Passes a batch through the encoder, backbone, and decoder"""
        x, y = batch

        if self.hparams.model.args.input_type == "raw":
            assert len(x.shape) == 3  # B,T,C
        elif self.hparams.model.args.input_type == "token":
            assert len(x.shape) == 2  # B,T

        pred = self.model.forward(x)

        return x, y, pred

    def training_step(self, batch, batch_idx):
        x, y, pred = self.forward(batch)
        loss = self.loss(pred, y)

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        record_step = {}
        for metric in self.hparams.metrics:
            if metric == "loss":
                record_step[f"trainer_{metric}"] = loss.item()
            # elif metric == "accuracy":
            #     # exclude ignore_index from accuracy calculation
            #     pred_class = pred.logits.argmax(dim=-1)
            #     record_step[f"trainer_{metric}"] = (
            #         (pred_class[y != -100] == y[y != -100]).float().mean().item()
            #     )

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
        x, y, pred = self.forward(batch)
        loss = self.loss(pred, y)

        record = {}
        for metric in self.hparams.metrics:
            if metric == "loss":
                record[f"validation_{metric}"] = loss.item()
            elif metric == "accuracy":
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

    def configure_optimizers(self):
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        param_groups = [{"params": self.model.parameters(), "lr": lr, "weight_decay": weight_decay}]
        optimizer = AdamW(param_groups)

        # Scheduler to mimic optax.cosine_onecycle_schedule
        # In PyTorch: OneCycleLR with annealing='cos' gives the cosine one-cycle shape.
        if self.hparams.lr_schedule:
            # Lightning can infer total steps: trainer.estimated_stepping_batches
            total_steps = self.hparams.total_steps or getattr(
                self.trainer, "estimated_stepping_batches", None
            )
            if total_steps is None:
                raise ValueError(
                    "total_steps not set. Pass total_steps=... to the module "
                    "or let Lightning set trainer.estimated_stepping_batches by calling trainer.fit first."
                )

            # For OneCycleLR we must pass max_lr per param group.
            # Use each group's current lr as its max_lr (the 'peak_value' from your optax code).
            max_lrs = [g.get("lr", lr) for g in optimizer.param_groups]

            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.hparams.pct_start,
                anneal_strategy="cos",
                cycle_momentum=False,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycleLR updates every step
                    "frequency": 1,
                },
            }


def create_trainer(config):
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups

        wandb_cfg = config.pop("wandb")
        logger = WandbLogger(
            config=OmegaConf.to_container(config, resolve=True),
            **wandb_cfg,
        )

    callbacks = []
    # checkpointing
    if config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.train.ckpt_dir,  # <--- where to save
                filename="epoch-{epoch:02d}-val_loss-{validation_loss:.2f}",
                save_top_k=3,  # how many best models to keep
                monitor="validation_loss",  # metric to monitor
                mode="min",  # minimize or maximize the metric
            )
        )
    if config.get("callbacks") is not None:
        for _name_, callback_config in config.callbacks.items():
            callback = getattr(src.callbacks, _name_)
            callbacks.append(callback(**callback_config))

    # Profiler
    profiler = None
    if config.trainer.get("profiler", None) is not None:
        profiler = hydra.utils.instantiate(config.trainer.profiler)
        config.trainer.pop("profiler")

    # Configure ddp automatically
    if config.trainer.accelerator == "gpu" and config.trainer.devices > 1:
        print("ddp automatically configured, more than 1 gpu used!")
        config.trainer.strategy = "ddp"

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        **config.trainer,
    )
    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    wrapper_model = SequenceLightningModule(config)
    summary = summarize(wrapper_model, max_depth=2)
    print(summary)
    print(f"Total parameters: {summary.total_parameters}")
    print(f"Trainable parameters: {summary.trainable_parameters}")

    data_module_class = get_data_module_class(config.dataset.pop("name"))
    data_module = data_module_class(**config.dataset)
    assert (
        config.task_type == data_module_class.SUPPORTED_TASK_TYPE
    ), f"Task type {config.task_type} not supported for dataset {config.dataset}"

    # Load pretrained_model if specified, UNTESTED
    if config.train.get("pretrained_model_path", None) is not None:
        # PTL style.  Note, method returns a new model object, and need to pass config.
        pretrained_model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )
        print("Loaded pretrained model from", config.train.pretrained_model_path)

        if config.train.get("ignore_pretrained_layers", False):
            pretrained_dict = pretrained_model.state_dict()
            model_dict = wrapper_model.state_dict()
            for k, v in model_dict.items():
                for ignore_layer in config.train.ignore_pretrained_layers:
                    if ignore_layer in k:
                        pretrained_dict[k] = v
            wrapper_model.load_state_dict(pretrained_dict)
        if config.train.get("pretrained_freeze_encoder", False):
            for name, param in wrapper_model.named_parameters():
                if not ("decoder" in name):
                    param.requires_grad = False

    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(wrapper_model, datamodule=data_module)

    if config.train.ckpt is not None:
        trainer.fit(wrapper_model, ckpt_path=config.train.ckpt, datamodule=data_module)
    else:
        trainer.fit(wrapper_model, datamodule=data_module)
    if config.train.test:
        trainer.test(wrapper_model, datamodule=data_module)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # sanity checks on task_type, loss, data config compatibility
    if cfg.task_type == "classification":
        assert cfg.loss == "cross_entropy"
    elif cfg.task_type == "generation":
        assert cfg.loss in ["mse", "cross_entropy"]

    # sanity check on loss, output_type, target_type compatibility
    if cfg.loss == "cross_entropy":
        assert cfg.model.args.output_type == "logits"
        assert cfg.dataset.target_type == "token"
    elif cfg.loss == "mse":
        assert cfg.model.args.output_type == "values"
        assert cfg.dataset.target_type == "raw"

    # check model input/output types and dataset input/target types are compatible
    assert cfg.model.args.input_type == cfg.dataset.input_type
    if cfg.model.args.output_type == "logits":
        assert cfg.dataset.target_type == "token"
    elif cfg.model.args.output_type == "values":
        assert cfg.dataset.target_type == "raw"

    # check preprocessing exists if output and input types are incompatible
    if (cfg.model.args.output_type == "logits" and cfg.model.args.input_type == "raw") or (
        cfg.model.args.output_type == "values" and cfg.model.args.input_type == "token"
    ):
        assert cfg.output2input_preprocess_fn_name is not None

    # Track with wandb
    wandb_cfg = cfg["wandb"]
    wandb.init(**wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True))

    train(cfg)


if __name__ == "__main__":
    main()
