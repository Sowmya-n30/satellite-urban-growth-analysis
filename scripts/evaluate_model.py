"""
Evaluate a trained model on the test set and report metrics.

Usage:
    python scripts/evaluate_model.py \
        --checkpoint checkpoints/unet_vijayawada/best_model.pth \
        --data data/processed/ \
        --model unet
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.unet import build_unet
from src.models.resnet import build_resnet50_segmenter
from src.models.efficientnet import build_efficientnet_segmenter
from src.data.dataloader import get_dataloaders
from src.training.metrics import SegmentationMetrics
from src.inference.predictor import UrbanPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Non-Urban", "Semi-Urban", "Urban"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained urban segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint .pth file")
    parser.add_argument("--data", type=str, default="data/processed",
                        help="Processed data directory")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "resnet50", "efficientnet"])
    parser.add_argument("--in-channels", type=int, default=15)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--output", type=str, default="results/reports/evaluation.json",
                        help="Save evaluation results to JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model_builders = {
        "unet": build_unet,
        "resnet50": build_resnet50_segmenter,
        "efficientnet": build_efficientnet_segmenter,
    }
    model = model_builders[args.model](
        in_channels=args.in_channels, num_classes=args.num_classes, pretrained=False
    )

    # Load checkpoint
    predictor = UrbanPredictor(
        model=model, device=device, num_classes=args.num_classes, use_tta=args.tta
    )
    UrbanPredictor.load_checkpoint(model, args.checkpoint, device=device)

    # Data
    _, _, test_loader, _ = get_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count() or 1),
    )

    # Evaluate
    metrics_tracker = SegmentationMetrics(num_classes=args.num_classes, device=device)

    for images, masks in test_loader:
        result = predictor.predict_batch(images)
        preds = torch.from_numpy(result["predictions"]).to(device)
        masks = masks.to(device)
        metrics_tracker.update(preds, masks)

    metrics = metrics_tracker.compute()
    metrics_tracker.print_report(class_names=CLASS_NAMES)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation results saved → %s", out_path)

    print(f"\n✅ Evaluation complete.")
    print(f"   Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"   Mean F1:  {metrics['mean_f1']:.4f}")
    print(f"   OA:       {metrics['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
