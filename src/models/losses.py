"""
Loss functions for satellite image segmentation and classification.

Provides:
    - DiceLoss: Region-overlap loss
    - FocalLoss: Down-weights easy examples
    - TverskyLoss: Generalised Dice with separate FP/FN weights
    - CombinedLoss: Weighted sum of Dice + CrossEntropy + Focal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.

    Args:
        smooth (float): Smoothing constant to avoid division by zero.
        ignore_index (int | None): Class index to ignore.
    """

    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): (B, C, H, W) raw class logits.
            targets (Tensor): (B, H, W) integer class labels.

        Returns:
            Tensor: Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dice = 0.0
        n_valid = 0
        for cls in range(num_classes):
            if cls == self.ignore_index:
                continue
            p = probs[:, cls]
            g = targets_one_hot[:, cls]
            intersection = (p * g).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice += (1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)).mean()
            n_valid += 1

        return dice / max(n_valid, 1)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma (float): Focusing parameter (default 2.0).
        alpha (float | None): Weighting factor for positive class.
        reduction (str): 'mean', 'sum', or 'none'.
        ignore_index (int): Class index to ignore.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean", ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): (B, C, H, W) or (B, C).
            targets (Tensor): (B, H, W) or (B,).

        Returns:
            Tensor: Focal loss scalar.
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss – generalised Dice that penalises FP and FN separately.

    Args:
        alpha (float): FP weight (default 0.3).
        beta (float): FN weight (default 0.7; penalise false negatives more).
        smooth (float): Smoothing constant.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        tversky = 0.0
        for cls in range(num_classes):
            p = probs[:, cls]
            g = targets_one_hot[:, cls]
            tp = (p * g).sum(dim=(1, 2))
            fp = (p * (1 - g)).sum(dim=(1, 2))
            fn = ((1 - p) * g).sum(dim=(1, 2))
            score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky += (1 - score).mean()

        return tversky / num_classes


class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice, CrossEntropy, and Focal losses.

    Args:
        dice_weight (float): Weight for Dice loss.
        ce_weight (float): Weight for CrossEntropy loss.
        focal_weight (float): Weight for Focal loss.
        class_weights (Tensor | None): Per-class weights for CrossEntropy.
        focal_gamma (float): Focal loss gamma.
        ignore_index (int): Label to ignore.
    """

    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.3,
        focal_weight=0.2,
        class_weights=None,
        focal_gamma=2.0,
        ignore_index=-100,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): (B, C, H, W) logits.
            targets (Tensor): (B, H, W) integer labels.

        Returns:
            tuple: (total_loss, dict of component losses)
        """
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets)
        focal_loss = self.focal(logits, targets)

        total = (
            self.dice_weight * dice_loss
            + self.ce_weight * ce_loss
            + self.focal_weight * focal_loss
        )
        components = {
            "dice": dice_loss.item(),
            "ce": ce_loss.item(),
            "focal": focal_loss.item(),
            "total": total.item(),
        }
        return total, components


def get_loss_function(loss_type="combined", class_weights=None, **kwargs):
    """
    Factory function to instantiate a loss function by name.

    Args:
        loss_type (str): One of 'dice', 'cross_entropy', 'focal', 'tversky', 'combined'.
        class_weights (Tensor | None): Per-class weights.
        **kwargs: Additional arguments passed to the loss constructor.

    Returns:
        nn.Module: Loss function instance.
    """
    loss_type = loss_type.lower()
    if loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, **kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(class_weights=class_weights, **kwargs)
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            "Choose from: dice, cross_entropy, focal, tversky, combined"
        )
