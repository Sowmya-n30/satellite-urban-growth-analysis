"""
Training callbacks for lifecycle management.

Provides:
    - EarlyStopping: Stop training when validation metric stops improving.
    - ModelCheckpoint: Save best and latest model checkpoints.
    - LearningRateLogger: Log LR changes.
"""

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience (int): Epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
        mode (str): 'min' (lower is better) or 'max' (higher is better).
    """

    def __init__(self, patience=15, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value) -> bool:
        """
        Check whether training should stop.

        Args:
            metric_value (float): Current epoch's monitored metric.

        Returns:
            bool: True if training should stop.
        """
        score = -metric_value if self.mode == "min" else metric_value

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d / %d", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Saves:
        - `best_model.pth` whenever validation loss improves.
        - `last_model.pth` at every epoch.

    Args:
        checkpoint_dir (str): Directory for checkpoint files.
        save_best (bool): Save when metric improves.
        mode (str): 'min' or 'max'.
        min_delta (float): Minimum improvement threshold.
        verbose (bool): Log checkpoint events.
    """

    def __init__(
        self,
        checkpoint_dir="checkpoints",
        save_best=True,
        mode="min",
        min_delta=1e-4,
        verbose=True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None

    def __call__(self, model, optimizer, epoch, metric_value):
        """
        Save checkpoints if appropriate.

        Args:
            model (nn.Module): Model to save.
            optimizer: Optimizer to save.
            epoch (int): Current epoch.
            metric_value (float): Monitored metric.
        """
        # Always save last checkpoint
        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metric": metric_value,
            },
            last_path,
        )

        # Save best checkpoint
        if self.save_best:
            score = -metric_value if self.mode == "min" else metric_value
            if self.best_score is None or score > self.best_score + self.min_delta:
                self.best_score = score
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metric": metric_value,
                    },
                    best_path,
                )
                if self.verbose:
                    logger.info(
                        "Checkpoint saved → %s (epoch=%d, metric=%.4f)",
                        best_path, epoch, metric_value,
                    )

    def load_best(self, model, optimizer=None, map_location=None):
        """
        Load the best checkpoint into model (and optionally optimizer).

        Args:
            model (nn.Module): Model to load weights into.
            optimizer: Optimizer to restore state (optional).
            map_location: torch.load map_location.

        Returns:
            int: Epoch at which the checkpoint was saved.
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_path}")

        checkpoint = torch.load(best_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        logger.info("Loaded best checkpoint (epoch=%d, metric=%.4f)", epoch, checkpoint["metric"])
        return epoch


class LearningRateLogger:
    """
    Callback to log learning rate changes each epoch.

    Args:
        optimizer: PyTorch optimizer.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.history = []

    def __call__(self, epoch):
        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.history.append((epoch, lrs))
        logger.info("Epoch %d | LR: %s", epoch, lrs)

    def get_lr(self):
        """Return current learning rates for all parameter groups."""
        return [pg["lr"] for pg in self.optimizer.param_groups]
