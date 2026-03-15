"""
Train a deep learning model for Vijayawada urban segmentation.

Supports: unet, resnet50, efficientnet, siamese (temporal change detection)

Usage:
    python scripts/train_model.py --model unet --data data/processed/ --epochs 100
    python scripts/train_model.py --model resnet50 --data data/processed/ --batch-size 16
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.unet import build_unet
from src.models.resnet import build_resnet50_segmenter
from src.models.efficientnet import build_efficientnet_segmenter
from src.models.temporal_model import build_siamese_model
from src.models.losses import get_loss_function
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_BUILDERS = {
    "unet": build_unet,
    "resnet50": build_resnet50_segmenter,
    "efficientnet": build_efficientnet_segmenter,
    "siamese": build_siamese_model,
}


def build_model(model_name, in_channels, num_classes):
    """Instantiate a model by name."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_name](in_channels=in_channels, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser(description="Train urban segmentation model")
    parser.add_argument("--model", type=str, default="unet", choices=list(MODEL_BUILDERS),
                        help="Model architecture")
    parser.add_argument("--data", type=str, default="data/processed",
                        help="Processed data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--in-channels", type=int, default=15,
                        help="Number of input channels (10 S2 bands + 5 indices)")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["dice", "cross_entropy", "focal", "combined"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    experiment_name = args.experiment or f"{args.model}_vijayawada"
    logger.info("Starting training: model=%s, experiment=%s", args.model, experiment_name)

    # Data
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count() or 1),
    )

    # Model
    model = build_model(args.model, args.in_channels, args.num_classes)

    # Loss
    class_weights = train_loader.dataset.get_class_weights() if hasattr(train_loader.dataset, "get_class_weights") else None
    loss_fn = get_loss_function(args.loss, class_weights=class_weights)

    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        num_classes=args.num_classes,
        device=args.device,
        mixed_precision=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=experiment_name,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
    )

    # Final test evaluation
    test_loss, test_metrics = trainer.validate(test_loader)
    logger.info("Test results: loss=%.4f, IoU=%.4f, F1=%.4f",
                test_loss, test_metrics["mean_iou"], test_metrics["mean_f1"])

    print(f"\n✅ Training complete!")
    print(f"   Best checkpoint: {args.checkpoint_dir}/{experiment_name}/best_model.pth")
    print(f"   Test IoU: {test_metrics['mean_iou']:.4f} | Test F1: {test_metrics['mean_f1']:.4f}")


if __name__ == "__main__":
    main()
