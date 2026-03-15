"""
Change detection analysis using Google Earth Engine for Vijayawada urban growth.

Implements multi-temporal change detection between two time periods:
    - Image differencing
    - NDBI-based urban change
    - CVA (Change Vector Analysis)

Usage:
    python gee_scripts/change_detection.py --year1 2020 --year2 2023
"""

import argparse
import logging
import sys

from gee_scripts.sentinel2_extract import (
    initialize_gee,
    get_vijayawada_roi,
    create_annual_composite,
    export_image_to_drive,
)
from gee_scripts.spectral_indices import add_all_indices, classify_urban

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_ndbi_change(composite_t1, composite_t2):
    """
    Compute NDBI difference between two composites.

    Positive values indicate new urban development (NDBI increased).
    Negative values indicate urban decline / revegetation.

    Args:
        composite_t1: ee.Image – earlier-date composite with NDBI band.
        composite_t2: ee.Image – later-date composite with NDBI band.

    Returns:
        ee.Image: NDBI change image named 'ndbi_change'.
    """
    ndbi_t1 = composite_t1.select("NDBI")
    ndbi_t2 = composite_t2.select("NDBI")
    change = ndbi_t2.subtract(ndbi_t1).rename("ndbi_change")
    return change


def compute_ndvi_change(composite_t1, composite_t2):
    """
    Compute NDVI difference between two composites.

    Negative values indicate vegetation loss (often due to urban expansion).

    Args:
        composite_t1: ee.Image – earlier-date composite with NDVI band.
        composite_t2: ee.Image – later-date composite with NDVI band.

    Returns:
        ee.Image: NDVI change image named 'ndvi_change'.
    """
    ndvi_t1 = composite_t1.select("NDVI")
    ndvi_t2 = composite_t2.select("NDVI")
    change = ndvi_t2.subtract(ndvi_t1).rename("ndvi_change")
    return change


def compute_urban_expansion(composite_t1, composite_t2):
    """
    Identify pixels that transitioned from non-urban to urban.

    Args:
        composite_t1: ee.Image – earlier-date composite.
        composite_t2: ee.Image – later-date composite.

    Returns:
        ee.Image: Binary mask (1 = new urban, 0 = no change) named 'urban_expansion'.
    """
    class_t1 = classify_urban(composite_t1)
    class_t2 = classify_urban(composite_t2)

    # New urban: was non-urban (0) or semi-urban (1), now urban (2)
    expansion = class_t2.eq(2).And(class_t1.lt(2)).rename("urban_expansion")
    return expansion


def compute_change_vector_analysis(composite_t1, composite_t2):
    """
    Change Vector Analysis (CVA) using NDBI and NDVI.

    Returns change magnitude and direction.

    Args:
        composite_t1: ee.Image – earlier composite.
        composite_t2: ee.Image – later composite.

    Returns:
        ee.Image: Multi-band image with 'change_magnitude' and 'change_direction'.
    """
    delta_ndbi = composite_t2.select("NDBI").subtract(composite_t1.select("NDBI"))
    delta_ndvi = composite_t2.select("NDVI").subtract(composite_t1.select("NDVI"))

    magnitude = delta_ndbi.pow(2).add(delta_ndvi.pow(2)).sqrt().rename("change_magnitude")
    direction = delta_ndbi.atan2(delta_ndvi).rename("change_direction")

    return magnitude.addBands(direction)


def run_change_detection(ee, year1=2020, year2=2023, folder="vijayawada_gee"):
    """
    Run the full change detection pipeline between two years.

    1. Create annual median composites for both years.
    2. Add spectral indices.
    3. Compute NDBI change, urban expansion, and CVA.
    4. Export results to Google Drive.

    Args:
        ee: Earth Engine module.
        year1: Earlier year.
        year2: Later year.
        folder: Google Drive folder.

    Returns:
        dict: Dictionary of export tasks.
    """
    roi = get_vijayawada_roi(ee)

    logger.info("Creating %d composite ...", year1)
    composite_t1 = add_all_indices(create_annual_composite(ee, roi, year1))

    logger.info("Creating %d composite ...", year2)
    composite_t2 = add_all_indices(create_annual_composite(ee, roi, year2))

    # --- Compute change layers ---
    ndbi_change = compute_ndbi_change(composite_t1, composite_t2)
    ndvi_change = compute_ndvi_change(composite_t1, composite_t2)
    urban_expansion = compute_urban_expansion(composite_t1, composite_t2)
    cva = compute_change_vector_analysis(composite_t1, composite_t2)

    # --- Export ---
    tasks = {}

    tasks["ndbi_change"] = export_image_to_drive(
        ee,
        ndbi_change,
        description=f"vijayawada_ndbi_change_{year1}_{year2}",
        folder=folder,
        roi=roi,
    )

    tasks["ndvi_change"] = export_image_to_drive(
        ee,
        ndvi_change,
        description=f"vijayawada_ndvi_change_{year1}_{year2}",
        folder=folder,
        roi=roi,
    )

    tasks["urban_expansion"] = export_image_to_drive(
        ee,
        urban_expansion,
        description=f"vijayawada_urban_expansion_{year1}_{year2}",
        folder=folder,
        roi=roi,
    )

    tasks["cva_magnitude"] = export_image_to_drive(
        ee,
        cva.select("change_magnitude"),
        description=f"vijayawada_change_magnitude_{year1}_{year2}",
        folder=folder,
        roi=roi,
    )

    logger.info(
        "Change detection tasks started for %d→%d. Check Google Drive > %s",
        year1,
        year2,
        folder,
    )
    return tasks


def compute_area_statistics(ee, image, roi, class_value, scale=30):
    """
    Compute area (in km²) of a given class in a classification image.

    Args:
        ee: Earth Engine module.
        image: ee.Image – classification image.
        roi: Region of interest.
        class_value: Integer class value to measure.
        scale: Pixel size in metres for area calculation.

    Returns:
        float: Area in km².
    """
    area_image = ee.Image.pixelArea().updateMask(image.eq(class_value))
    area = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=scale,
        maxPixels=1e13,
    )
    area_sqm = area.get("area").getInfo()
    return area_sqm / 1e6  # Convert m² to km²


def main():
    parser = argparse.ArgumentParser(
        description="Urban change detection for Vijayawada using GEE"
    )
    parser.add_argument(
        "--year1", type=int, default=2020, help="Start year (default: 2020)"
    )
    parser.add_argument(
        "--year2", type=int, default=2023, help="End year (default: 2023)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="vijayawada_gee",
        help="Google Drive folder for exports",
    )
    args = parser.parse_args()

    ee = initialize_gee()
    run_change_detection(ee, year1=args.year1, year2=args.year2, folder=args.folder)

    print(
        f"\n✅ Change detection ({args.year1}→{args.year2}) export tasks started."
    )
    print("   Monitor at: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
