from pathlib import Path

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
import src.data
import src.tasks


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
                dirpath=Path(config.train.ckpt_dir)
                / wandb.run.project
                / f"{wandb.run.name}-{wandb.run.id}",  # <--- where to save
                filename="epoch-{epoch:02d}-val_loss-{validation_loss:.2f}",
                save_top_k=3,  # how many best models to keep
                monitor="validation_loss",  # metric to monitor
                mode="min",  # minimize or maximize the metric
            )
        )
    if config.get("callbacks") is not None:
        for _name_, callback_config in config.callbacks.items():
            callback = getattr(src.callbacks, _name_)
            if "sample_dir" in callback_config:
                callback_config["sample_dir"] = (
                    Path(callback_config["sample_dir"])
                    / wandb.run.project
                    / f"{wandb.run.name}-{wandb.run.id}"
                )
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
    task = getattr(src.tasks, config.task.pop("name"))(config)
    summary = summarize(task, max_depth=2)
    print(summary)
    print(f"Total parameters: {summary.total_parameters}")
    print(f"Trainable parameters: {summary.trainable_parameters}")

    for name, p in task.model.named_parameters():
        print(f"{name:55s} shape={tuple(p.shape)}  requires_grad={p.requires_grad}")

    data_module = getattr(src.data, config.dataset.pop("name"))(**config.dataset)

    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(task, datamodule=data_module)

    if config.train.ckpt is not None:
        trainer.fit(task, ckpt_path=config.train.ckpt, datamodule=data_module)
    else:
        trainer.fit(task, datamodule=data_module)
    if config.train.test:
        trainer.test(task, datamodule=data_module)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # sanity check on loss, target_type compatibility
    if cfg.task.loss == "cross_entropy":
        assert cfg.task.target_type == "token"
    elif cfg.task.loss == "mse":
        assert cfg.task.target_type == "value"

    # Track with wandb
    wandb_cfg = cfg["wandb"]
    wandb.init(**wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True))

    train(cfg)


if __name__ == "__main__":
    main()
