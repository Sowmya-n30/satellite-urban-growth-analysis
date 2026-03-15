"""
Data augmentation transforms for satellite imagery using Albumentations.

Provides:
    - get_train_transforms: Heavy augmentation for training.
    - get_val_transforms:   Light preprocessing for validation / inference.

All transforms work on numpy images of shape (H, W, C) with float32 values.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    logger.warning("albumentations not installed. Augmentation disabled. Run: pip install albumentations")
    ALBUMENTATIONS_AVAILABLE = False


def get_train_transforms(patch_size=256):
    """
    Build training augmentation pipeline for satellite patches.

    Includes spatial and spectral augmentations suitable for
    multi-spectral imagery.

    Args:
        patch_size (int): Target patch size after random crop.

    Returns:
        A.Compose transform or None if albumentations is unavailable.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose(
        [
            # Spatial augmentations
            A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),

            # Radiometric / spectral augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GridDistortion(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),

            # Coarse dropout for robustness
            A.CoarseDropout(
                max_holes=8,
                max_height=patch_size // 16,
                max_width=patch_size // 16,
                fill_value=0,
                p=0.2,
            ),
        ],
        additional_targets={"image2": "image"},
    )


def get_val_transforms(patch_size=256):
    """
    Build validation / test preprocessing pipeline (no augmentation).

    Args:
        patch_size (int): Target patch size.

    Returns:
        A.Compose transform or None.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose(
        [
            A.CenterCrop(height=patch_size, width=patch_size, p=1.0),
        ],
        additional_targets={"image2": "image"},
    )


def get_tta_transforms():
    """
    Test-Time Augmentation (TTA) transforms.

    Returns a list of transforms; predictions are averaged.

    Returns:
        list[A.Compose]: 8-fold TTA (original + 3 rotations + 4 flips).
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return [None]

    transforms = [
        A.Compose([]),                          # Original
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0), A.HorizontalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0), A.VerticalFlip(p=1.0)]),
        A.Compose([A.Transpose(p=1.0)]),
    ]
    return transforms
