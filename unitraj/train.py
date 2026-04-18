import os
import hydra
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint  # Import ModelCheckpoint

from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint

date_time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
torch.set_float32_matmul_precision("medium")


class EpochSummary(Callback):
    """Print one compact epoch summary when SLURM disables tqdm."""

    metric_names = (
        "train/brier_fde",
        "val/brier_fde",
        "val/minADE6",
        "val/minFDE6",
        "val/miss_rate",
    )

    def _format_metric(self, metrics, metric_name):
        value = metrics.get(metric_name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return f"{metric_name}={value:.4f}"

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metric_strs = [
            self._format_metric(trainer.callback_metrics, metric_name)
            for metric_name in self.metric_names
        ]
        metric_strs = [metric for metric in metric_strs if metric is not None]
        if not metric_strs:
            return

        print("\n", flush=True)  # Ensure the epoch summary is printed on a new line
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} | " + " | ".join(metric_strs),
            flush=True,
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    exp_dir = Path(cfg.exp_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []
    call_backs.append(EpochSummary())

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=str(checkpoint_dir)
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '_'  # Use '-' instead of '=' in checkpoint filenames

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(
            project="unitraj",
            name=cfg.exp_name,
            id=f"{cfg.exp_name}_{date_time_now}",
            save_dir=str(exp_dir),
        ),
        default_root_dir=str(exp_dir),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs,
        enable_progress_bar=os.environ.get("SLURM_JOB_ID") is None,
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        search_pattern = str(checkpoint_dir / '**' / '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
