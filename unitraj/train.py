import os
import hydra
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint

from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint

date_time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
torch.set_float32_matmul_precision('medium')

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

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=str(checkpoint_dir)
    )

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
        enable_progress_bar=True if os.environ.get("SLURM_JOB_ID") is None else False,  # Disable progress bar in SLURM environment
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        search_pattern = str(checkpoint_dir / '**' / '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
