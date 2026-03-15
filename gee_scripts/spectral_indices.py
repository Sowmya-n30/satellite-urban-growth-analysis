"""
Spectral index calculation utilities for Sentinel-2 imagery in Google Earth Engine.

Supported indices:
    - NDVI  : Normalized Difference Vegetation Index
    - NDBI  : Normalized Difference Built-up Index
    - NDMI  : Normalized Difference Moisture Index
    - MNDWI : Modified Normalized Difference Water Index
    - EVI   : Enhanced Vegetation Index
    - BSI   : Bare Soil Index

Usage:
    from gee_scripts.spectral_indices import add_all_indices
    image_with_indices = add_all_indices(sentinel2_image)
"""

import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Per-image index functions (designed for .map() on an ImageCollection)
# ------------------------------------------------------------------


def add_ndvi(image):
    """
    Add NDVI band to a Sentinel-2 image.

    NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
    Range: [-1, 1]. Urban areas typically < 0.1; vegetation > 0.4.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def add_ndbi(image):
    """
    Add NDBI band to a Sentinel-2 image.

    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) = (B11 - B8) / (B11 + B8)
    Range: [-1, 1]. Built-up areas typically > 0.1.
    """
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands(ndbi)


def add_ndmi(image):
    """
    Add NDMI band to a Sentinel-2 image.

    NDMI = (NIR - SWIR1) / (NIR + SWIR1) = (B8 - B11) / (B8 + B11)
    Range: [-1, 1]. Water/moisture > 0.
    """
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return image.addBands(ndmi)


def add_mndwi(image):
    """
    Add MNDWI band to a Sentinel-2 image.

    MNDWI = (Green - SWIR1) / (Green + SWIR1) = (B3 - B11) / (B3 + B11)
    Range: [-1, 1]. Water bodies > 0.
    """
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    return image.addBands(mndwi)


def add_evi(image):
    """
    Add EVI band to a Sentinel-2 image.

    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    Reduces atmospheric and soil background effects.
    """
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2"),
        },
    ).rename("EVI")
    return image.addBands(evi)


def add_bsi(image):
    """
    Add BSI (Bare Soil Index) band to a Sentinel-2 image.

    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    High BSI indicates bare soil or impervious surfaces.
    """
    bsi = image.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {
            "SWIR1": image.select("B11"),
            "RED": image.select("B4"),
            "NIR": image.select("B8"),
            "BLUE": image.select("B2"),
        },
    ).rename("BSI")
    return image.addBands(bsi)


def add_all_indices(image):
    """
    Add all spectral indices (NDVI, NDBI, NDMI, MNDWI, EVI, BSI) to a Sentinel-2 image.

    Args:
        image: ee.Image with Sentinel-2 bands (B2, B3, B4, B8, B11).

    Returns:
        ee.Image with original bands plus index bands appended.
    """
    image = add_ndvi(image)
    image = add_ndbi(image)
    image = add_ndmi(image)
    image = add_mndwi(image)
    image = add_evi(image)
    image = add_bsi(image)
    return image


# ------------------------------------------------------------------
# Urban classification thresholds
# ------------------------------------------------------------------

URBAN_THRESHOLDS = {
    "ndbi_min": 0.0,       # NDBI > 0 → built-up
    "ndvi_max": 0.2,       # NDVI < 0.2 → non-vegetated
    "mndwi_max": -0.1,     # MNDWI < -0.1 → not water
    "semi_ndbi_min": -0.1, # -0.1 < NDBI < 0 → semi-urban
}


def classify_urban(image):
    """
    Produce a simple 3-class urban classification map.

    Classes:
        0 = Non-Urban (vegetation, bare soil)
        1 = Semi-Urban (mixed / transitional)
        2 = Urban (built-up, impervious surfaces)

    Args:
        image: ee.Image with NDBI, NDVI, MNDWI bands already added.

    Returns:
        ee.Image: Single-band classification image with values 0/1/2.
    """
    ndbi = image.select("NDBI")
    ndvi = image.select("NDVI")
    mndwi = image.select("MNDWI")

    water = mndwi.gt(URBAN_THRESHOLDS["mndwi_max"])
    urban = (
        ndbi.gt(URBAN_THRESHOLDS["ndbi_min"])
        .And(ndvi.lt(URBAN_THRESHOLDS["ndvi_max"]))
        .And(water.Not())
    )
    semi_urban = (
        ndbi.gt(URBAN_THRESHOLDS["semi_ndbi_min"])
        .And(ndbi.lte(URBAN_THRESHOLDS["ndbi_min"]))
        .And(water.Not())
    )

    classification = (
        urban.multiply(2)
        .add(semi_urban.multiply(1))
        .rename("classification")
    )
    return classification


def export_indices(ee, image, roi, year, folder="vijayawada_gee"):
    """
    Export individual spectral index GeoTIFFs to Google Drive.

    Args:
        ee: Earth Engine module.
        image: ee.Image with index bands added.
        roi: Export region geometry.
        year: Year label for filenames.
        folder: Google Drive folder name.
    """
    indices = ["NDVI", "NDBI", "NDMI", "MNDWI", "EVI", "BSI"]
    tasks = []
    for idx in indices:
        task = ee.batch.Export.image.toDrive(
            image=image.select(idx),
            description=f"vijayawada_{idx.lower()}_{year}",
            folder=folder,
            scale=10,
            region=roi,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        task.start()
        logger.info("Export task started: vijayawada_%s_%d", idx.lower(), year)
        tasks.append(task)
    return tasks
