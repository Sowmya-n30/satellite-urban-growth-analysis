"""
Plotting utilities for training metrics, confusion matrices, and spectral analysis.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Install: pip install matplotlib")

CLASS_NAMES = ["Non-Urban", "Semi-Urban", "Urban"]
CLASS_COLORS = ["#2ECC71", "#F39C12", "#E74C3C"]


def _require_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required. Install: pip install matplotlib")


def plot_training_history(history: dict, output_path: str = None):
    """
    Plot training/validation loss and metrics over epochs.

    Args:
        history (dict): Keys include 'train_loss', 'val_loss', 'val_iou', 'val_f1'.
        output_path (str | None): Save path; if None, displays interactively.
    """
    _require_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history.get("train_loss", []), label="Train", color="steelblue")
    ax.plot(epochs, history.get("val_loss", []), label="Val", color="tomato")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IoU
    ax = axes[1]
    ax.plot(epochs, history.get("val_iou", []), color="mediumseagreen")
    ax.set_title("Validation Mean IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # F1
    ax = axes[2]
    ax.plot(epochs, history.get("val_f1", []), color="darkorange")
    ax.set_title("Validation Mean F1")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_confusion_matrix(cm: np.ndarray, class_names=None, output_path: str = None):
    """
    Plot a normalized confusion matrix.

    Args:
        cm (np.ndarray): (num_classes, num_classes) confusion matrix.
        class_names (list | None): Class name labels.
        output_path (str | None): Save path.
    """
    _require_matplotlib()

    class_names = class_names or CLASS_NAMES
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_classification_map(prediction: np.ndarray, title="Classification Map", output_path: str = None):
    """
    Visualize a classification map with the project color scheme.

    Args:
        prediction (np.ndarray): (H, W) integer class map.
        title (str): Plot title.
        output_path (str | None): Save path.
    """
    _require_matplotlib()

    cmap = mcolors.ListedColormap(CLASS_COLORS)
    bounds_map = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds_map, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(prediction, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(CLASS_NAMES)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_change_heatmap(change_map: np.ndarray, title="Urban Change 2020→2023", output_path: str = None):
    """
    Plot a change detection heatmap.

    Args:
        change_map (np.ndarray): (H, W) float change map.
        title (str): Plot title.
        output_path (str | None): Save path.
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(change_map, cmap="RdYlGn_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="NDBI Change", fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_spectral_bands(image: np.ndarray, band_names=None, output_path: str = None):
    """
    Visualize individual spectral bands of a multi-spectral image.

    Args:
        image (np.ndarray): (H, W, C) image.
        band_names (list | None): Band labels.
        output_path (str | None): Save path.
    """
    _require_matplotlib()

    n_bands = image.shape[-1]
    band_names = band_names or [f"Band {i}" for i in range(n_bands)]
    cols = min(5, n_bands)
    rows = (n_bands + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.array(axes).flatten()

    for i in range(n_bands):
        band = image[:, :, i]
        p2, p98 = np.percentile(band, [2, 98])
        axes[i].imshow(np.clip(band, p2, p98), cmap="gray")
        axes[i].set_title(band_names[i], fontsize=9)
        axes[i].axis("off")

    for i in range(n_bands, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sentinel-2 Spectral Bands", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_rgb(image: np.ndarray, title="RGB Image", output_path: str = None):
    """
    Display an RGB composite from a multi-spectral image (uses B4/B3/B2 indices 2,1,0).

    Args:
        image (np.ndarray): (H, W, C) image array (bands B2=0, B3=1, B4=2).
        title (str): Plot title.
        output_path (str | None): Save path.
    """
    _require_matplotlib()

    rgb = image[:, :, [2, 1, 0]].copy()  # B4, B3, B2
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _save_or_show(fig, output_path)


def _save_or_show(fig, output_path):
    """Save figure to file or display it."""
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Plot saved → %s", out)
        plt.close(fig)
    else:
        plt.show()
