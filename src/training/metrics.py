"""
Evaluation metrics for semantic segmentation and classification.

Implements:
    - Per-class and mean IoU (Intersection over Union)
    - Per-class and mean Dice coefficient
    - F1 score, Precision, Recall
    - Confusion matrix
    - OA (Overall Accuracy)
"""

import logging
from typing import Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Accumulate per-batch predictions and compute segmentation metrics.

    Args:
        num_classes (int): Number of segmentation classes.
        device (str | torch.device): Computation device.
        ignore_index (int | None): Class index to ignore.
    """

    def __init__(self, num_classes: int = 3, device="cpu", ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=self.device
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with a batch of predictions.

        Args:
            preds (Tensor): (B, H, W) predicted class indices.
            targets (Tensor): (B, H, W) ground-truth class indices.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            preds = preds[valid]
            targets = targets[valid]

        # Accumulate confusion matrix
        mask = (targets >= 0) & (targets < self.num_classes)
        combined = self.num_classes * targets[mask] + preds[mask]
        bincount = torch.bincount(combined, minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            dict: Keys include per-class and mean metrics.
        """
        cm = self.confusion_matrix.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        tn = cm.sum() - tp - fp - fn

        # IoU per class
        iou = tp / (tp + fp + fn + 1e-8)

        # Dice per class
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        # Precision & Recall per class
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # F1 per class
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Overall accuracy
        oa = tp.sum() / (cm.sum() + 1e-8)

        metrics = {
            "per_class_iou": iou.cpu().numpy().tolist(),
            "per_class_dice": dice.cpu().numpy().tolist(),
            "per_class_precision": precision.cpu().numpy().tolist(),
            "per_class_recall": recall.cpu().numpy().tolist(),
            "per_class_f1": f1.cpu().numpy().tolist(),
            "mean_iou": iou.mean().item(),
            "mean_dice": dice.mean().item(),
            "mean_precision": precision.mean().item(),
            "mean_recall": recall.mean().item(),
            "mean_f1": f1.mean().item(),
            "overall_accuracy": oa.item(),
        }
        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix as a numpy array."""
        return self.confusion_matrix.cpu().numpy()

    def print_report(self, class_names=None):
        """Print a human-readable metrics report."""
        metrics = self.compute()
        class_names = class_names or [f"Class {i}" for i in range(self.num_classes)]

        print("\n" + "=" * 60)
        print(f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
        print("-" * 60)
        for i, name in enumerate(class_names):
            print(
                f"{name:<20} "
                f"{metrics['per_class_iou'][i]:>8.4f} "
                f"{metrics['per_class_dice'][i]:>8.4f} "
                f"{metrics['per_class_f1'][i]:>8.4f} "
                f"{metrics['per_class_precision'][i]:>8.4f} "
                f"{metrics['per_class_recall'][i]:>8.4f}"
            )
        print("-" * 60)
        print(
            f"{'Mean':<20} "
            f"{metrics['mean_iou']:>8.4f} "
            f"{metrics['mean_dice']:>8.4f} "
            f"{metrics['mean_f1']:>8.4f} "
            f"{metrics['mean_precision']:>8.4f} "
            f"{metrics['mean_recall']:>8.4f}"
        )
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
        print("=" * 60)


def compute_iou_numpy(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute per-class IoU from flat numpy arrays.

    Args:
        pred (np.ndarray): Predicted labels, shape (N,).
        target (np.ndarray): True labels, shape (N,).
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Per-class IoU, shape (num_classes,).
    """
    iou = np.zeros(num_classes)
    for cls in range(num_classes):
        tp = np.logical_and(pred == cls, target == cls).sum()
        fp = np.logical_and(pred == cls, target != cls).sum()
        fn = np.logical_and(pred != cls, target == cls).sum()
        iou[cls] = tp / (tp + fp + fn + 1e-8)
    return iou
