"""
Preprocess raw GeoTIFF satellite imagery into ML-ready .npy patch datasets.

Pipeline:
    1. Read GeoTIFF files from data/raw/
    2. Normalize / standardize bands
    3. Compute spectral indices
    4. Extract patches
    5. Split into train/val/test sets
    6. Save to data/processed/

Usage:
    python scripts/preprocess_data.py --input data/raw/ --output data/processed/ --patch-size 256
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import (
    read_geotiff,
    normalize_image,
    add_spectral_indices,
    extract_patches,
    save_patches,
    split_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_single_image(image_path, mask_path, output_dir, patch_size, stride):
    """Process one image-mask pair into patches."""
    logger.info("Processing: %s", image_path)

    image, profile = read_geotiff(image_path)
    mask, _ = read_geotiff(mask_path)

    if mask.ndim == 3:
        mask = mask[:, :, 0]  # Take first band for mask

    # Normalize and add spectral indices
    image = normalize_image(image)
    image = add_spectral_indices(image)

    # Extract patches
    stem = Path(image_path).stem
    image_patches, mask_patches = extract_patches(image, mask.astype("int64"), patch_size=patch_size, stride=stride)

    # Save patches
    save_patches(image_patches, mask_patches, output_dir, prefix=stem)

    return len(image_patches)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Sentinel-2 GeoTIFF data into patch datasets"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw",
        help="Input directory with GeoTIFF files"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Output directory for processed patches"
    )
    parser.add_argument(
        "--mask-dir", type=str, default=None,
        help="Directory with mask GeoTIFFs (defaults to input/masks/)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=256,
        help="Patch size in pixels"
    )
    parser.add_argument(
        "--stride", type=int, default=128,
        help="Stride between patches"
    )
    parser.add_argument(
        "--skip-split", action="store_true",
        help="Skip train/val/test split (only extract patches)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    mask_dir = Path(args.mask_dir) if args.mask_dir else input_dir / "masks"
    patches_dir = output_dir / "patches"

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    image_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not image_files:
        logger.warning("No GeoTIFF files found in %s. Skipping preprocessing.", input_dir)
        logger.info("Place your GeoTIFF files in %s to preprocess them.", input_dir)
        return

    total_patches = 0
    for image_path in image_files:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            logger.warning("No mask found for %s, skipping.", image_path.name)
            continue
        n = process_single_image(
            str(image_path), str(mask_path), str(patches_dir),
            args.patch_size, args.stride
        )
        total_patches += n

    logger.info("Total patches extracted: %d", total_patches)

    if not args.skip_split and total_patches > 0:
        logger.info("Splitting into train/val/test ...")
        split_dataset(
            image_dir=patches_dir / "images",
            mask_dir=patches_dir / "masks",
            output_dir=output_dir,
        )
        logger.info("Dataset split complete. Output: %s", output_dir)

    print(f"\n✅ Preprocessing complete. {total_patches} patches saved to {output_dir}")


if __name__ == "__main__":
    main()
