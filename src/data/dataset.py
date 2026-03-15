"""
PyTorch Dataset classes for multi-spectral satellite imagery.

Supports:
    - Patch-based loading from GeoTIFF files
    - On-the-fly spectral index computation
    - Multi-temporal data for change detection
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Non-Urban", 1: "Semi-Urban", 2: "Urban"}
NUM_CLASSES = 3


class SatelliteDataset(Dataset):
    """
    Dataset for single-date multi-spectral satellite image patches.

    Directory layout expected:
        root_dir/
            images/  *.npy  (H x W x C float32 arrays, bands in [0,1])
            masks/   *.npy  (H x W int64 arrays, values in {0,1,2})

    Args:
        root_dir (str | Path): Root directory containing images/ and masks/.
        transform: Optional albumentations transform applied to both image and mask.
        normalize (bool): If True, z-score normalise each band using dataset stats.
        stats (dict | None): Precomputed mean/std per band {'mean': array, 'std': array}.
    """

    def __init__(self, root_dir, transform=None, normalize=True, stats=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.normalize = normalize
        self.stats = stats

        self.image_paths = sorted(
            glob.glob(str(self.root_dir / "images" / "*.npy"))
        )
        self.mask_paths = sorted(
            glob.glob(str(self.root_dir / "masks" / "*.npy"))
        )

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Mismatch: {len(self.image_paths)} images vs "
                f"{len(self.mask_paths)} masks in {root_dir}"
            )
        if len(self.image_paths) == 0:
            raise ValueError(f"No .npy files found in {root_dir}/images/")

        logger.info("Loaded dataset from %s: %d samples", root_dir, len(self))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)

        if self.normalize and self.stats is not None:
            mean = self.stats["mean"].reshape(1, 1, -1)
            std = self.stats["std"].reshape(1, 1, -1)
            image = (image - mean) / (std + 1e-8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors: image (C, H, W), mask (H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor

    @property
    def num_channels(self):
        """Number of spectral channels in the dataset."""
        sample = np.load(self.image_paths[0])
        return sample.shape[-1]

    def compute_stats(self):
        """
        Compute per-band mean and std across all patches.

        Returns:
            dict: {'mean': np.ndarray shape (C,), 'std': np.ndarray shape (C,)}
        """
        logger.info("Computing dataset statistics ...")
        running_sum = None
        running_sq = None
        count = 0

        for path in self.image_paths:
            img = np.load(path).astype(np.float64)  # (H, W, C)
            h, w, c = img.shape
            if running_sum is None:
                running_sum = np.zeros(c)
                running_sq = np.zeros(c)
            running_sum += img.sum(axis=(0, 1))
            running_sq += (img ** 2).sum(axis=(0, 1))
            count += h * w

        mean = running_sum / count
        std = np.sqrt(running_sq / count - mean ** 2)
        logger.info("Dataset stats computed: mean=%s, std=%s", mean, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def get_class_weights(self):
        """
        Compute inverse-frequency class weights for imbalanced segmentation.

        Returns:
            torch.Tensor: Weights tensor of shape (NUM_CLASSES,).
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for path in self.mask_paths:
            mask = np.load(path)
            for cls in range(NUM_CLASSES):
                counts[cls] += (mask == cls).sum()

        total = counts.sum()
        weights = total / (NUM_CLASSES * counts + 1e-8)
        weights = weights / weights.sum()
        logger.info("Class weights: %s", weights)
        return torch.tensor(weights, dtype=torch.float32)


class TemporalSatelliteDataset(Dataset):
    """
    Dataset for multi-temporal change detection.

    Directory layout:
        root_dir/
            t1/images/  *.npy  – earlier-date patches
            t2/images/  *.npy  – later-date patches
            masks/      *.npy  – change masks (0=no change, 1=changed)

    Args:
        root_dir (str | Path): Root directory.
        transform: Optional albumentations transform.
        stats (dict | None): Normalisation stats with keys 't1' and 't2'.
    """

    def __init__(self, root_dir, transform=None, stats=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.stats = stats or {}

        self.t1_paths = sorted(
            glob.glob(str(self.root_dir / "t1" / "images" / "*.npy"))
        )
        self.t2_paths = sorted(
            glob.glob(str(self.root_dir / "t2" / "images" / "*.npy"))
        )
        self.mask_paths = sorted(
            glob.glob(str(self.root_dir / "masks" / "*.npy"))
        )

        if not (len(self.t1_paths) == len(self.t2_paths) == len(self.mask_paths)):
            raise ValueError(
                "Mismatch in number of t1/t2/mask files in %s" % root_dir
            )
        logger.info("Temporal dataset loaded: %d pairs", len(self))

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx):
        img_t1 = np.load(self.t1_paths[idx]).astype(np.float32)
        img_t2 = np.load(self.t2_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)

        if "t1" in self.stats:
            mean = self.stats["t1"]["mean"].reshape(1, 1, -1)
            std = self.stats["t1"]["std"].reshape(1, 1, -1)
            img_t1 = (img_t1 - mean) / (std + 1e-8)

        if "t2" in self.stats:
            mean = self.stats["t2"]["mean"].reshape(1, 1, -1)
            std = self.stats["t2"]["std"].reshape(1, 1, -1)
            img_t2 = (img_t2 - mean) / (std + 1e-8)

        if self.transform is not None:
            # Apply identical spatial transforms to both images and mask
            augmented = self.transform(image=img_t1, image2=img_t2, mask=mask)
            img_t1 = augmented["image"]
            img_t2 = augmented["image2"]
            mask = augmented["mask"]

        t1_tensor = torch.from_numpy(img_t1).permute(2, 0, 1).float()
        t2_tensor = torch.from_numpy(img_t2).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        return t1_tensor, t2_tensor, mask_tensor


class ClassificationDataset(Dataset):
    """
    Image-level classification dataset for ResNet/EfficientNet.

    Args:
        image_paths (list[str]): Paths to .npy patch files.
        labels (list[int]): Corresponding integer class labels.
        transform: Optional albumentations transform.
        stats (dict | None): Normalisation stats.
    """

    def __init__(self, image_paths, labels, transform=None, stats=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.stats = stats

        assert len(image_paths) == len(labels), "image_paths and labels must have the same length"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).astype(np.float32)
        label = self.labels[idx]

        if self.stats is not None:
            mean = self.stats["mean"].reshape(1, 1, -1)
            std = self.stats["std"].reshape(1, 1, -1)
            image = (image - mean) / (std + 1e-8)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return image_tensor, torch.tensor(label, dtype=torch.long)
