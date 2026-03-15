"""
Preprocessing utilities for Sentinel-2 GeoTIFF imagery.

Handles:
    - Reading multi-band GeoTIFF files
    - Cloud masking
    - Band normalization
    - Spectral index computation (NumPy-based, no GEE required)
    - Patch extraction for deep-learning training
    - Train / val / test dataset splitting
    - PCA for dimensionality reduction
"""

import os
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    logger.warning("rasterio not installed. File I/O disabled. Run: pip install rasterio")
    RASTERIO_AVAILABLE = False

# Band indices within the 10-band Sentinel-2 stack [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
BAND_IDX = {
    "B2": 0, "B3": 1, "B4": 2, "B5": 3,
    "B6": 4, "B7": 5, "B8": 6, "B8A": 7,
    "B11": 8, "B12": 9,
}


# ------------------------------------------------------------------
# File I/O
# ------------------------------------------------------------------

def read_geotiff(path):
    """
    Read a multi-band GeoTIFF into a (H, W, C) float32 numpy array.

    Args:
        path (str | Path): Path to the GeoTIFF file.

    Returns:
        tuple: (image np.ndarray shape HxWxC, profile dict)
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required. Install with: pip install rasterio")
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # (C, H, W)
        profile = src.profile
    image = np.moveaxis(data, 0, -1)            # → (H, W, C)
    return image, profile


def write_geotiff(path, data, profile):
    """
    Write a (H, W) or (H, W, C) numpy array to a GeoTIFF.

    Args:
        path (str | Path): Output file path.
        data (np.ndarray): Image array (H, W) or (H, W, C).
        profile (dict): rasterio profile from the source image.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required.")
    if data.ndim == 2:
        data = data[..., np.newaxis]
    c, h, w = data.shape[2], data.shape[0], data.shape[1]
    profile.update(count=c, dtype=data.dtype)
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(c):
            dst.write(data[:, :, i], i + 1)


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

def normalize_image(image, clip_percentile=2):
    """
    Normalize a (H, W, C) float32 image to [0, 1] per band.

    Uses percentile clipping to handle outliers.

    Args:
        image (np.ndarray): Input image (H, W, C).
        clip_percentile (float): Percentile for low/high clipping.

    Returns:
        np.ndarray: Normalized image in [0, 1].
    """
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[-1]):
        band = image[:, :, c]
        lo = np.percentile(band, clip_percentile)
        hi = np.percentile(band, 100 - clip_percentile)
        out[:, :, c] = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def standardize_image(image, mean=None, std=None):
    """
    Z-score standardize a (H, W, C) image per band.

    Args:
        image (np.ndarray): Input (H, W, C).
        mean (np.ndarray | None): Per-band mean of shape (C,).
        std (np.ndarray | None): Per-band std of shape (C,).

    Returns:
        np.ndarray: Standardized image.
    """
    if mean is None:
        mean = image.mean(axis=(0, 1))
    if std is None:
        std = image.std(axis=(0, 1))
    return (image - mean) / (std + 1e-8)


# ------------------------------------------------------------------
# Spectral Indices (NumPy)
# ------------------------------------------------------------------

def compute_ndvi(image):
    """NDVI = (B8 - B4) / (B8 + B4). Returns (H, W) array."""
    b8 = image[:, :, BAND_IDX["B8"]]
    b4 = image[:, :, BAND_IDX["B4"]]
    return (b8 - b4) / (b8 + b4 + 1e-8)


def compute_ndbi(image):
    """NDBI = (B11 - B8) / (B11 + B8). Returns (H, W) array."""
    b11 = image[:, :, BAND_IDX["B11"]]
    b8 = image[:, :, BAND_IDX["B8"]]
    return (b11 - b8) / (b11 + b8 + 1e-8)


def compute_ndmi(image):
    """NDMI = (B8 - B11) / (B8 + B11). Returns (H, W) array."""
    b8 = image[:, :, BAND_IDX["B8"]]
    b11 = image[:, :, BAND_IDX["B11"]]
    return (b8 - b11) / (b8 + b11 + 1e-8)


def compute_mndwi(image):
    """MNDWI = (B3 - B11) / (B3 + B11). Returns (H, W) array."""
    b3 = image[:, :, BAND_IDX["B3"]]
    b11 = image[:, :, BAND_IDX["B11"]]
    return (b3 - b11) / (b3 + b11 + 1e-8)


def compute_evi(image):
    """EVI = 2.5*(B8-B4)/(B8+6*B4-7.5*B2+1). Returns (H, W) array."""
    b8 = image[:, :, BAND_IDX["B8"]]
    b4 = image[:, :, BAND_IDX["B4"]]
    b2 = image[:, :, BAND_IDX["B2"]]
    return 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + 1e-8)


def add_spectral_indices(image):
    """
    Append NDVI, NDBI, NDMI, MNDWI, EVI bands to a (H, W, C) image.

    Returns:
        np.ndarray: Shape (H, W, C+5).
    """
    ndvi = compute_ndvi(image)[..., np.newaxis]
    ndbi = compute_ndbi(image)[..., np.newaxis]
    ndmi = compute_ndmi(image)[..., np.newaxis]
    mndwi = compute_mndwi(image)[..., np.newaxis]
    evi = compute_evi(image)[..., np.newaxis]
    return np.concatenate([image, ndvi, ndbi, ndmi, mndwi, evi], axis=-1)


# ------------------------------------------------------------------
# Patch extraction
# ------------------------------------------------------------------

def extract_patches(image, mask, patch_size=256, stride=128):
    """
    Extract overlapping patches from a large satellite image.

    Args:
        image (np.ndarray): (H, W, C) image array.
        mask (np.ndarray): (H, W) label mask.
        patch_size (int): Patch side length.
        stride (int): Stride between patches.

    Returns:
        tuple: (image_patches list, mask_patches list)
    """
    h, w = image.shape[:2]
    image_patches, mask_patches = [], []

    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            img_patch = image[row:row + patch_size, col:col + patch_size, :]
            msk_patch = mask[row:row + patch_size, col:col + patch_size]
            image_patches.append(img_patch)
            mask_patches.append(msk_patch)

    logger.info("Extracted %d patches (%dx%d, stride=%d)", len(image_patches), patch_size, patch_size, stride)
    return image_patches, mask_patches


def save_patches(image_patches, mask_patches, output_dir, prefix="patch"):
    """
    Save extracted patches as .npy files.

    Args:
        image_patches (list[np.ndarray]): Image patches.
        mask_patches (list[np.ndarray]): Mask patches.
        output_dir (str | Path): Destination directory.
        prefix (str): Filename prefix.
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    msk_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, msk) in enumerate(zip(image_patches, mask_patches)):
        np.save(img_dir / f"{prefix}_{i:05d}.npy", img)
        np.save(msk_dir / f"{prefix}_{i:05d}.npy", msk)

    logger.info("Saved %d patches to %s", len(image_patches), output_dir)


# ------------------------------------------------------------------
# Train / val / test split
# ------------------------------------------------------------------

def split_dataset(
    image_dir,
    mask_dir,
    output_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
):
    """
    Split patch .npy files into train / val / test sets.

    Args:
        image_dir (str | Path): Directory of image .npy patches.
        mask_dir (str | Path): Directory of mask .npy patches.
        output_dir (str | Path): Root output directory.
        train_ratio (float): Fraction for training.
        val_ratio (float): Fraction for validation.
        seed (int): Random seed.
    """
    import shutil

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    image_files = sorted(image_dir.glob("*.npy"))
    mask_files = sorted(mask_dir.glob("*.npy"))
    assert len(image_files) == len(mask_files), "Image/mask count mismatch"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(image_files))

    n = len(indices)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split, idxs in splits.items():
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        for idx in idxs:
            shutil.copy(image_files[idx], output_dir / split / "images" / image_files[idx].name)
            shutil.copy(mask_files[idx], output_dir / split / "masks" / mask_files[idx].name)
        logger.info("Split '%s': %d samples", split, len(idxs))


# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------

def apply_pca(image, n_components=6):
    """
    Apply PCA to reduce spectral dimensionality.

    Args:
        image (np.ndarray): (H, W, C) image.
        n_components (int): Number of PCA components to keep.

    Returns:
        np.ndarray: (H, W, n_components) reduced image.
    """
    from sklearn.decomposition import PCA

    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(pixels)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA: %d → %d components, explained variance=%.2f%%",
        c, n_components, explained * 100,
    )
    return transformed.reshape(h, w, n_components)
