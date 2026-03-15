"""
Post-processing utilities for segmentation predictions.

Includes:
    - Morphological operations (opening, closing)
    - Small region removal
    - GeoTIFF export of predictions
    - GeoJSON polygon export
    - Urban growth statistics computation
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed. Morphological operations disabled.")

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

CLASS_NAMES = {0: "Non-Urban", 1: "Semi-Urban", 2: "Urban"}


def remove_small_regions(prediction: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove small isolated regions from a classification map.

    Args:
        prediction (np.ndarray): (H, W) integer class map.
        min_area (int): Minimum number of pixels for a valid region.

    Returns:
        np.ndarray: Cleaned prediction map.
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy required for remove_small_regions.")
        return prediction

    cleaned = prediction.copy()
    for cls in np.unique(prediction):
        if cls == 0:
            continue
        binary = (prediction == cls).astype(np.uint8)
        labeled, n_components = ndimage.label(binary)
        component_sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
        for idx, size in enumerate(component_sizes):
            if size < min_area:
                cleaned[labeled == (idx + 1)] = 0

    return cleaned


def apply_morphological_ops(prediction: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Apply morphological closing to fill small holes and smooth boundaries.

    Args:
        prediction (np.ndarray): (H, W) integer class map.
        iterations (int): Number of dilation/erosion iterations.

    Returns:
        np.ndarray: Smoothed prediction.
    """
    if not SCIPY_AVAILABLE:
        return prediction

    struct = ndimage.generate_binary_structure(2, 2)
    result = np.zeros_like(prediction)

    for cls in np.unique(prediction):
        binary = (prediction == cls).astype(np.uint8)
        closed = ndimage.binary_closing(binary, structure=struct, iterations=iterations)
        result[closed > 0] = cls

    return result


def save_prediction_geotiff(
    prediction: np.ndarray,
    output_path: str,
    reference_profile: dict = None,
    bbox: tuple = None,
    crs_epsg: int = 4326,
):
    """
    Save a prediction map as a GeoTIFF.

    Args:
        prediction (np.ndarray): (H, W) integer class map.
        output_path (str): Output file path.
        reference_profile (dict | None): rasterio profile from the source image.
        bbox (tuple | None): (min_lon, min_lat, max_lon, max_lat) if no reference profile.
        crs_epsg (int): Coordinate reference system EPSG code.
    """
    if not RASTERIO_AVAILABLE:
        logger.error("rasterio required for GeoTIFF export.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = prediction.shape

    if reference_profile is not None:
        profile = reference_profile.copy()
        profile.update(count=1, dtype="uint8")
    else:
        if bbox is None:
            bbox = (80.4, 16.3, 80.9, 16.8)  # Default Vijayawada bbox
        transform = from_bounds(*bbox, w, h)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": w,
            "height": h,
            "count": 1,
            "crs": CRS.from_epsg(crs_epsg),
            "transform": transform,
            "compress": "lzw",
        }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction.astype("uint8"), 1)

    logger.info("Saved prediction GeoTIFF → %s", output_path)


def compute_urban_statistics(
    prediction: np.ndarray,
    pixel_area_m2: float = 100.0,  # 10m × 10m = 100 m²
) -> dict:
    """
    Compute urban land-cover statistics from a prediction map.

    Args:
        prediction (np.ndarray): (H, W) integer class map.
        pixel_area_m2 (float): Area of one pixel in square metres.

    Returns:
        dict: Per-class area in km², percentage, and pixel count.
    """
    total_pixels = prediction.size
    total_area_km2 = total_pixels * pixel_area_m2 / 1e6

    stats = {"total_area_km2": total_area_km2}
    for cls, name in CLASS_NAMES.items():
        count = int((prediction == cls).sum())
        area_km2 = count * pixel_area_m2 / 1e6
        pct = count / total_pixels * 100
        stats[name] = {
            "pixel_count": count,
            "area_km2": round(area_km2, 3),
            "percentage": round(pct, 2),
        }

    return stats


def compute_change_statistics(
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    pixel_area_m2: float = 100.0,
) -> dict:
    """
    Compute urban growth / change statistics between two prediction maps.

    Args:
        pred_t1 (np.ndarray): Earlier classification map.
        pred_t2 (np.ndarray): Later classification map.
        pixel_area_m2 (float): Pixel area in m².

    Returns:
        dict: Change statistics including new urban area and transition matrix.
    """
    assert pred_t1.shape == pred_t2.shape, "Prediction maps must have the same shape."

    stats_t1 = compute_urban_statistics(pred_t1, pixel_area_m2)
    stats_t2 = compute_urban_statistics(pred_t2, pixel_area_m2)

    urban_t1_km2 = stats_t1["Urban"]["area_km2"]
    urban_t2_km2 = stats_t2["Urban"]["area_km2"]
    growth_km2 = urban_t2_km2 - urban_t1_km2
    growth_pct = growth_km2 / (urban_t1_km2 + 1e-8) * 100

    # New urban pixels (non-urban → urban)
    new_urban = ((pred_t1 < 2) & (pred_t2 == 2)).sum()
    new_urban_km2 = new_urban * pixel_area_m2 / 1e6

    return {
        "urban_area_t1_km2": urban_t1_km2,
        "urban_area_t2_km2": urban_t2_km2,
        "urban_growth_km2": round(growth_km2, 3),
        "urban_growth_percent": round(growth_pct, 2),
        "new_urban_pixels": int(new_urban),
        "new_urban_area_km2": round(new_urban_km2, 3),
        "t1_stats": stats_t1,
        "t2_stats": stats_t2,
    }
