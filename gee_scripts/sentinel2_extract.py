"""
Sentinel-2 data extraction from Google Earth Engine for Vijayawada urban analysis.

This script connects to GEE and downloads Sentinel-2 imagery for Vijayawada,
Andhra Pradesh, India, across multiple years for change detection.

Usage:
    python gee_scripts/sentinel2_extract.py --year 2023 --output data/raw/

Requirements:
    pip install earthengine-api geemap
    Run `earthengine authenticate` before using.
"""

import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Vijayawada region of interest
VIJAYAWADA_CENTER = [16.5062, 80.6480]
VIJAYAWADA_BBOX = [80.4, 16.3, 80.9, 16.8]
BUFFER_RADIUS_M = 30000  # 30 km

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_THRESHOLD = 20

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_NAMES = {
    "B2": "Blue_490nm",
    "B3": "Green_560nm",
    "B4": "Red_665nm",
    "B5": "RedEdge1_705nm",
    "B6": "RedEdge2_740nm",
    "B7": "RedEdge3_783nm",
    "B8": "NIR_842nm",
    "B8A": "NarrowNIR_865nm",
    "B11": "SWIR1_1610nm",
    "B12": "SWIR2_2190nm",
}


def initialize_gee():
    """Authenticate and initialize Google Earth Engine."""
    try:
        import ee
        ee.Initialize()
        logger.info("GEE initialized successfully.")
        return ee
    except Exception:
        try:
            import ee
            ee.Authenticate()
            ee.Initialize()
            logger.info("GEE authenticated and initialized.")
            return ee
        except ImportError:
            logger.error("earthengine-api not installed. Run: pip install earthengine-api")
            sys.exit(1)
        except Exception as exc:
            logger.error("GEE initialization failed: %s", exc)
            sys.exit(1)


def get_vijayawada_roi(ee):
    """Return the Region of Interest (ROI) geometry for Vijayawada."""
    point = ee.Geometry.Point(VIJAYAWADA_CENTER[::-1])  # GEE uses [lon, lat]
    roi = point.buffer(BUFFER_RADIUS_M)
    return roi


def mask_clouds_s2(image):
    """Apply cloud masking using the Sentinel-2 QA60 band."""
    import ee
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000).copyProperties(
        image, ["system:time_start"]
    )


def get_sentinel2_collection(ee, roi, start_date, end_date):
    """
    Fetch a cloud-filtered Sentinel-2 collection for the given ROI and date range.

    Returns:
        ee.ImageCollection: Filtered Sentinel-2 image collection.
    """
    collection = (
        ee.ImageCollection(SENTINEL2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
        .select(BANDS + ["QA60"])
        .map(mask_clouds_s2)
    )
    count = collection.size().getInfo()
    logger.info(
        "Found %d images between %s and %s", count, start_date, end_date
    )
    return collection


def create_annual_composite(ee, roi, year):
    """
    Create a median annual composite for the given year.

    Args:
        ee: Earth Engine module.
        roi: Region of interest geometry.
        year: Integer year.

    Returns:
        ee.Image: Median composite image clipped to ROI.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    collection = get_sentinel2_collection(ee, roi, start_date, end_date)
    composite = collection.select(BANDS).median().clip(roi)
    logger.info("Created %d annual median composite.", year)
    return composite


def export_image_to_drive(ee, image, description, folder, roi, scale=10):
    """
    Export an ee.Image to Google Drive as a GeoTIFF.

    Args:
        ee: Earth Engine module.
        image: ee.Image to export.
        description: Export task description / filename.
        folder: Google Drive folder name.
        roi: Export region geometry.
        scale: Pixel resolution in metres.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=scale,
        region=roi,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    logger.info("Export task '%s' started – check Google Drive > %s", description, folder)
    return task


def export_multispectral_bands(ee, roi, year, folder="vijayawada_gee"):
    """
    Export all Sentinel-2 bands for the given year to Google Drive.

    Args:
        ee: Earth Engine module.
        roi: Region of interest.
        year: Year for the composite.
        folder: Google Drive folder name.
    """
    composite = create_annual_composite(ee, roi, year)

    export_image_to_drive(
        ee,
        composite,
        description=f"vijayawada_S2_multispectral_{year}",
        folder=folder,
        roi=roi,
    )

    rgb = composite.select(["B4", "B3", "B2"])
    export_image_to_drive(
        ee,
        rgb,
        description=f"vijayawada_S2_rgb_{year}",
        folder=folder,
        roi=roi,
    )

    logger.info("Export tasks started for year %d.", year)


def export_all_years(ee, roi, years=None, folder="vijayawada_gee"):
    """
    Export Sentinel-2 composites for multiple years.

    Args:
        ee: Earth Engine module.
        roi: Region of interest.
        years: List of integer years (default 2020-2023).
        folder: Google Drive folder name.
    """
    if years is None:
        years = [2020, 2021, 2022, 2023]
    tasks = []
    for year in years:
        composite = create_annual_composite(ee, roi, year)
        task = export_image_to_drive(
            ee,
            composite,
            description=f"vijayawada_S2_{year}",
            folder=folder,
            roi=roi,
        )
        tasks.append(task)
    logger.info("Queued %d export tasks.", len(tasks))
    return tasks


def print_collection_info(ee, roi, year):
    """Print summary statistics for the given year's image collection."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    collection = get_sentinel2_collection(ee, roi, start_date, end_date)
    count = collection.size().getInfo()
    print(f"\nYear {year}: {count} cloud-free Sentinel-2 images available")
    if count > 0:
        first = collection.first()
        props = first.propertyNames().getInfo()
        date = first.date().format("YYYY-MM-dd").getInfo()
        print(f"  First image date: {date}")
        print(f"  Properties: {props[:5]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Sentinel-2 data for Vijayawada from GEE"
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023],
        help="Year(s) to export (default: 2020 2021 2022 2023)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="vijayawada_gee",
        help="Google Drive folder name for exports",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print collection info without exporting",
    )
    args = parser.parse_args()

    ee = initialize_gee()
    roi = get_vijayawada_roi(ee)

    if args.info_only:
        for year in args.year:
            print_collection_info(ee, roi, year)
        return

    export_all_years(ee, roi, years=args.year, folder=args.folder)
    print("\n✅ Export tasks started. Check Google Drive > %s" % args.folder)
    print("   Monitor progress at: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
