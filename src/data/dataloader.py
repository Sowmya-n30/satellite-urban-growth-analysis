"""
DataLoader factory for satellite image datasets.

Creates train/val/test DataLoader objects with appropriate settings for
multi-spectral segmentation and classification tasks.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import SatelliteDataset, TemporalSatelliteDataset, ClassificationDataset
from src.data.augmentation import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def get_dataloaders(
    data_dir,
    batch_size=16,
    num_workers=4,
    patch_size=256,
    normalize=True,
    pin_memory=True,
    stats=None,
):
    """
    Create train, validation, and test DataLoaders for the segmentation task.

    Expected directory structure:
        data_dir/
            train/images/, train/masks/
            val/images/,   val/masks/
            test/images/,  test/masks/

    Args:
        data_dir (str): Root data directory.
        batch_size (int): Batch size.
        num_workers (int): DataLoader worker processes.
        patch_size (int): Spatial size of image patches.
        normalize (bool): Whether to normalise inputs.
        pin_memory (bool): Pin memory for faster GPU transfer.
        stats (dict | None): Pre-computed normalisation stats.

    Returns:
        tuple: (train_loader, val_loader, test_loader, stats)
    """
    data_dir = Path(data_dir)

    train_transform = get_train_transforms(patch_size=patch_size)
    val_transform = get_val_transforms(patch_size=patch_size)

    train_dataset = SatelliteDataset(
        root_dir=data_dir / "train",
        transform=train_transform,
        normalize=normalize,
        stats=stats,
    )

    # Compute stats from training set if not provided
    if normalize and stats is None:
        stats = train_dataset.compute_stats()
        train_dataset.stats = stats
        logger.info("Computed normalisation stats from training set.")

    val_dataset = SatelliteDataset(
        root_dir=data_dir / "val",
        transform=val_transform,
        normalize=normalize,
        stats=stats,
    )
    test_dataset = SatelliteDataset(
        root_dir=data_dir / "test",
        transform=val_transform,
        normalize=normalize,
        stats=stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "DataLoaders: train=%d, val=%d, test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader, stats


def get_temporal_dataloaders(
    data_dir,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
):
    """
    Create DataLoaders for the temporal change-detection task.

    Args:
        data_dir (str): Root directory with t1/, t2/, masks/ sub-directories.
        batch_size (int): Batch size.
        num_workers (int): Worker processes.
        pin_memory (bool): Pin memory.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    dataset = TemporalSatelliteDataset(root_dir=data_dir)
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
