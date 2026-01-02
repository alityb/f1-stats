"""
Trainer class for pace decomposition model.

Handles training loop, validation, checkpointing, and logging.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.loss import PaceDecompositionLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for pace decomposition model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.loss_fn = PaceDecompositionLoss(config)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Logging
        self.use_tensorboard = config.training.logging.use_tensorboard
        if self.use_tensorboard:
            log_dir = Path(config.training.logging.log_dir) / "runs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.epochs_without_improvement = 0

        # Checkpointing
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _create_optimizer(self):
        """Create optimizer from config."""
        opt_config = self.config.training.optimizer

        if opt_config.type == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.type}")

        logger.info(f"Optimizer: {opt_config.type}, lr={opt_config.lr}")
        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        sched_config = self.config.training.scheduler

        if sched_config.type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.mode,
                factor=sched_config.factor,
                patience=sched_config.patience,
                min_lr=sched_config.min_lr,
            )
        else:
            scheduler = None

        logger.info(f"Scheduler: {sched_config.type}")
        return scheduler

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {"pred_loss": 0.0, "emb_norm_loss": 0.0, "driver_shrink_loss": 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            output = self.model(batch)

            # Compute loss
            loss_dict = self.loss_fn(output, batch, self.model)
            loss = loss_dict["total_loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += loss_dict[key].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log to TensorBoard
            if self.writer and self.global_step % self.config.training.logging.log_every_n_batches == 0:
                self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)

            self.global_step += 1

        # Average metrics over epoch
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()

        val_loss = 0.0
        val_metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            output = self.model(batch)

            # Compute loss
            loss_dict = self.loss_fn(output, batch, self.model)
            val_loss += loss_dict["total_loss"].item()

            # Compute metrics
            metrics = self.loss_fn.compute_metrics(output, batch)
            for key in val_metrics:
                val_metrics[key] += metrics[key]

            num_batches += 1

        # Average over validation set
        avg_val_loss = val_loss / num_batches
        avg_val_metrics = {k: v / num_batches for k, v in val_metrics.items()}

        return avg_val_loss, avg_val_metrics

    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)

        num_epochs = self.config.training.epochs
        early_stop_patience = self.config.training.early_stopping.patience
        eval_every = self.config.training.validation.eval_every_n_epochs

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_metrics = self.train_epoch()

            logger.info(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            logger.info(f"  Pred Loss: {train_metrics['pred_loss']:.4f}")
            logger.info(f"  Emb Norm: {train_metrics['emb_norm_loss']:.4f}")
            logger.info(f"  Driver Shrink: {train_metrics['driver_shrink_loss']:.4f}")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)

            # Validation
            if (epoch + 1) % eval_every == 0:
                val_loss, val_metrics = self.validate()

                logger.info(f"\nEpoch {epoch} - Validation")
                logger.info(f"  Loss: {val_loss:.4f}")
                logger.info(f"  RMSE: {val_metrics['rmse']:.4f}s")
                logger.info(f"  MAE: {val_metrics['mae']:.4f}s")

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, epoch)
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f"val/{key}", value, epoch)

                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_metrics['rmse'])
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"  Learning rate: {current_lr:.6f}")

                # Checkpointing
                monitor_metric = val_metrics['rmse']
                if monitor_metric < self.best_val_metric:
                    self.best_val_metric = monitor_metric
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(f"best_model.pt", is_best=True)
                    logger.info(f"  ✅ New best model! RMSE: {monitor_metric:.4f}s")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.config.training.early_stopping.enabled:
                    if self.epochs_without_improvement >= early_stop_patience:
                        logger.info(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                        break

            # Periodic checkpointing
            if (epoch + 1) % self.config.training.checkpoint.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        logger.info("\n" + "=" * 80)
        logger.info("Training Complete")
        logger.info("=" * 80)
        logger.info(f"Best validation RMSE: {self.best_val_metric:.4f}s")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            logger.info(f"  Saved best model to {checkpoint_path}")
        else:
            logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_metric = checkpoint["best_val_metric"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {self.current_epoch}, Best Val Metric: {self.best_val_metric:.4f}")
