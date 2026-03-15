"""
Generate urban classification maps and change detection reports.

Usage:
    python scripts/generate_maps.py \
        --checkpoint checkpoints/unet_vijayawada/best_model.pth \
        --image data/raw/vijayawada_S2_2023.tif \
        --image-t1 data/raw/vijayawada_S2_2020.tif \
        --image-t2 data/raw/vijayawada_S2_2023.tif \
        --output results/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.models.unet import build_unet
from src.data.preprocessing import read_geotiff, normalize_image, add_spectral_indices
from src.inference.predictor import UrbanPredictor
from src.inference.postprocessing import (
    apply_morphological_ops,
    remove_small_regions,
    save_prediction_geotiff,
    compute_urban_statistics,
    compute_change_statistics,
)
from src.visualization.plots import plot_classification_map, plot_change_heatmap, plot_rgb
from src.visualization.reports import generate_html_report, generate_markdown_report, save_statistics_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default Vijayawada bounding box [[S, W], [N, E]]
VIJAYAWADA_BOUNDS = [[16.3, 80.4], [16.8, 80.9]]


def load_and_predict(image_path, model, device, stats=None):
    """Load a GeoTIFF and run sliding-window inference."""
    image, profile = read_geotiff(image_path)
    image = normalize_image(image)
    image = add_spectral_indices(image)

    predictor = UrbanPredictor(model=model, device=device, use_tta=False)
    result = predictor.predict_large_image(image, patch_size=256, stride=128, stats=stats)

    # Post-process
    pred = result["prediction"]
    pred = remove_small_regions(pred, min_area=50)
    pred = apply_morphological_ops(pred)

    return pred, result["confidence"], profile


def main():
    parser = argparse.ArgumentParser(description="Generate urban classification maps")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None,
                        help="Single image for classification map")
    parser.add_argument("--image-t1", type=str, default=None,
                        help="Earlier image for change detection")
    parser.add_argument("--image-t2", type=str, default=None,
                        help="Later image for change detection")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--in-channels", type=int, default=15)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_unet(in_channels=args.in_channels, num_classes=args.num_classes)
    UrbanPredictor.load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    if args.image:
        logger.info("Running single-image classification ...")
        pred, conf, profile = load_and_predict(args.image, model, device)

        # Save GeoTIFF
        save_prediction_geotiff(
            pred,
            output_path=str(output_dir / "predictions" / "classification.tif"),
            reference_profile=profile,
        )

        # Visualization
        plot_classification_map(
            pred,
            title="Vijayawada Urban Classification",
            output_path=str(output_dir / "visualizations" / "classification_map.png"),
        )

        stats = compute_urban_statistics(pred)
        save_statistics_json(stats, str(output_dir / "reports" / "urban_statistics.json"))

        logger.info("Urban statistics: %s", stats)

    if args.image_t1 and args.image_t2:
        logger.info("Running change detection ...")
        pred_t1, _, profile = load_and_predict(args.image_t1, model, device)
        pred_t2, _, _ = load_and_predict(args.image_t2, model, device)

        change_stats = compute_change_statistics(pred_t1, pred_t2)

        # Change heatmap
        change_map = (pred_t2 == 2).astype(float) - (pred_t1 == 2).astype(float)
        plot_change_heatmap(
            change_map,
            title="Urban Growth 2020→2023",
            output_path=str(output_dir / "visualizations" / "change_detection.png"),
        )

        # Reports
        generate_html_report(change_stats, output_path=str(output_dir / "reports" / "urban_growth_report.html"))
        generate_markdown_report(change_stats, output_path=str(output_dir / "reports" / "urban_growth_report.md"))
        save_statistics_json(change_stats, str(output_dir / "reports" / "change_statistics.json"))

        print(f"\n📊 Urban Growth Summary:")
        print(f"   2020 Urban Area: {change_stats['urban_area_t1_km2']} km²")
        print(f"   2023 Urban Area: {change_stats['urban_area_t2_km2']} km²")
        print(f"   Growth: +{change_stats['urban_growth_km2']} km² ({change_stats['urban_growth_percent']}%)")

    print(f"\n✅ Map generation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
