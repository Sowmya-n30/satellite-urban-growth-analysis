"""
Training loop for satellite image segmentation and classification.

Supports:
    - Mixed-precision (AMP) training
    - Learning rate scheduling
    - Tensorboard and CSV logging
    - Checkpoint saving (best and latest)
    - Gradient clipping
"""

import os
import csv
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.training.metrics import SegmentationMetrics
from src.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    General-purpose trainer for semantic segmentation models.

    Args:
        model (nn.Module): PyTorch model.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function returning (loss, components_dict).
        scheduler: LR scheduler (optional).
        num_classes (int): Number of segmentation classes.
        device (str | torch.device): 'cuda', 'cpu', or 'mps'.
        mixed_precision (bool): Enable AMP training.
        gradient_clip (float | None): Max gradient norm.
        checkpoint_dir (str): Directory for saving checkpoints.
        log_dir (str): Directory for Tensorboard / CSV logs.
        experiment_name (str): Sub-directory name for this run.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        num_classes=3,
        device="cuda",
        mixed_precision=True,
        gradient_clip=1.0,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        experiment_name="unet_vijayawada",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.gradient_clip = gradient_clip
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = SegmentationMetrics(num_classes=num_classes, device=self.device)
        self._init_tensorboard()
        self._init_csv_log()

        logger.info(
            "Trainer ready | device=%s | mixed_precision=%s",
            self.device, self.mixed_precision,
        )

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            self.writer = None
            logger.warning("TensorBoard not available. Install tensorboard for logging.")

    def _init_csv_log(self):
        self.csv_path = self.log_dir / "training_log.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "val_loss",
                    "val_iou", "val_dice", "val_f1",
                    "val_precision", "val_recall", "lr",
                ])

    def _log_csv(self, row):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def train_one_epoch(self, loader, epoch):
        """Run one training epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss_output = self.loss_fn(logits, masks)
                # Support both scalar loss and (loss, components) tuple
                if isinstance(loss_output, tuple):
                    loss, components = loss_output
                else:
                    loss = loss_output
                    components = {}

            self.scaler.scale(loss).backward()

            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % max(1, n_batches // 10) == 0:
                logger.info(
                    "Epoch %d [%d/%d] loss=%.4f",
                    epoch, batch_idx, n_batches, loss.item(),
                )

        avg_loss = total_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, loader):
        """Run validation and return loss + segmentation metrics."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss_output = self.loss_fn(logits, masks)
                if isinstance(loss_output, tuple):
                    loss, _ = loss_output
                else:
                    loss = loss_output

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            self.metrics.update(preds, masks)

        avg_loss = total_loss / len(loader)
        metric_dict = self.metrics.compute()
        return avg_loss, metric_dict

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_best=True,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs (int): Maximum epochs.
            early_stopping_patience (int): Early stopping patience.
            save_best (bool): Save checkpoint when val loss improves.

        Returns:
            dict: Training history.
        """
        early_stopper = EarlyStopping(patience=early_stopping_patience)
        checkpointer = ModelCheckpoint(
            checkpoint_dir=str(self.checkpoint_dir),
            save_best=save_best,
        )

        history = {
            "train_loss": [], "val_loss": [],
            "val_iou": [], "val_dice": [], "val_f1": [],
        }

        for epoch in range(1, epochs + 1):
            t_start = time.time()

            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, metrics = self.validate(val_loader)

            elapsed = time.time() - t_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "IoU=%.4f | Dice=%.4f | F1=%.4f | lr=%.2e | %.1fs",
                epoch, epochs, train_loss, val_loss,
                metrics["mean_iou"], metrics["mean_dice"], metrics["mean_f1"],
                current_lr, elapsed,
            )

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(metrics["mean_iou"])
            history["val_dice"].append(metrics["mean_dice"])
            history["val_f1"].append(metrics["mean_f1"])

            # Tensorboard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/IoU", metrics["mean_iou"], epoch)
                self.writer.add_scalar("Metrics/Dice", metrics["mean_dice"], epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # CSV log
            self._log_csv([
                epoch, train_loss, val_loss,
                metrics["mean_iou"], metrics["mean_dice"], metrics["mean_f1"],
                metrics.get("mean_precision", 0), metrics.get("mean_recall", 0),
                current_lr,
            ])

            # Learning rate scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            # Checkpoint
            checkpointer(self.model, self.optimizer, epoch, val_loss)

            # Early stopping
            if early_stopper(val_loss):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        if self.writer:
            self.writer.close()

        logger.info("Training complete. Best val_loss=%.4f", early_stopper.best_score)
        return history
