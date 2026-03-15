"""
Download Sentinel-2 data for Vijayawada from Google Earth Engine.

Usage:
    python scripts/download_data.py --years 2020 2021 2022 2023 --folder vijayawada_gee
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gee_scripts.sentinel2_extract import initialize_gee, get_vijayawada_roi, export_all_years
from gee_scripts.spectral_indices import add_all_indices, export_indices
from gee_scripts.sentinel2_extract import create_annual_composite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download Vijayawada Sentinel-2 data from GEE"
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023],
        help="Years to download"
    )
    parser.add_argument(
        "--folder", type=str, default="vijayawada_gee",
        help="Google Drive folder"
    )
    parser.add_argument(
        "--indices-only", action="store_true",
        help="Export only spectral indices (NDVI, NDBI, etc.)"
    )
    args = parser.parse_args()

    logger.info("Initializing Google Earth Engine ...")
    ee = initialize_gee()
    roi = get_vijayawada_roi(ee)

    if args.indices_only:
        logger.info("Exporting spectral indices for years: %s", args.years)
        for year in args.years:
            composite = create_annual_composite(ee, roi, year)
            composite_with_indices = add_all_indices(composite)
            export_indices(ee, composite_with_indices, roi, year, folder=args.folder)
    else:
        logger.info("Exporting multi-spectral data for years: %s", args.years)
        export_all_years(ee, roi, years=args.years, folder=args.folder)

    print("\n✅ Export tasks queued. Monitor at: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
