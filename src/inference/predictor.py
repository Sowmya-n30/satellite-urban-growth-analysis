"""
Inference / prediction module for satellite image segmentation.

Supports:
    - Single-image inference
    - Batch inference with sliding window
    - Test-Time Augmentation (TTA)
    - Confidence score maps
    - Large GeoTIFF tiled inference
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.augmentation import get_tta_transforms

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Non-Urban", 1: "Semi-Urban", 2: "Urban"}
CLASS_COLORS = {
    0: (46, 204, 113),    # Green
    1: (243, 156, 18),    # Orange
    2: (231, 76, 60),     # Red
}


class UrbanPredictor:
    """
    Run inference on satellite image patches.

    Args:
        model (nn.Module): Trained segmentation model.
        device (str | torch.device): Inference device.
        num_classes (int): Number of segmentation classes.
        use_tta (bool): Apply Test-Time Augmentation.
        confidence_threshold (float): Minimum confidence for a valid prediction.
    """

    def __init__(
        self,
        model,
        device="cuda",
        num_classes=3,
        use_tta=False,
        confidence_threshold=0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.num_classes = num_classes
        self.use_tta = use_tta
        self.confidence_threshold = confidence_threshold
        self.tta_transforms = get_tta_transforms() if use_tta else [None]

        logger.info(
            "Predictor ready | device=%s | TTA=%s", self.device, use_tta
        )

    @torch.no_grad()
    def predict_patch(self, image: np.ndarray) -> dict:
        """
        Predict on a single (H, W, C) numpy patch.

        Args:
            image (np.ndarray): Float32 patch, shape (H, W, C).

        Returns:
            dict with keys:
                - 'prediction': (H, W) int class map
                - 'probabilities': (H, W, num_classes) probability map
                - 'confidence': (H, W) max-probability map
        """
        prob_accum = None

        for transform in self.tta_transforms:
            if transform is not None:
                aug = transform(image=image)
                img = aug["image"]
            else:
                img = image

            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            tensor = tensor.to(self.device)

            logits = self.model(tensor)          # (1, C, H, W)
            probs = F.softmax(logits, dim=1)     # (1, C, H, W)
            probs_np = probs.squeeze(0).cpu().numpy()  # (C, H, W)
            probs_np = np.moveaxis(probs_np, 0, -1)    # (H, W, C)

            if prob_accum is None:
                prob_accum = probs_np
            else:
                prob_accum += probs_np

        prob_accum /= len(self.tta_transforms)

        prediction = prob_accum.argmax(axis=-1).astype(np.int32)
        confidence = prob_accum.max(axis=-1)

        # Set low-confidence pixels to background (class 0)
        prediction[confidence < self.confidence_threshold] = 0

        return {
            "prediction": prediction,
            "probabilities": prob_accum,
            "confidence": confidence,
        }

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> dict:
        """
        Predict on a batch of image tensors.

        Args:
            images (Tensor): (B, C, H, W) batch.

        Returns:
            dict: 'predictions' (B, H, W), 'probabilities' (B, H, W, C).
        """
        images = images.to(self.device)
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        return {
            "predictions": preds.cpu().numpy(),
            "probabilities": probs.permute(0, 2, 3, 1).cpu().numpy(),
        }

    def predict_large_image(
        self,
        image: np.ndarray,
        patch_size: int = 256,
        stride: int = 128,
        stats: dict = None,
    ) -> dict:
        """
        Sliding-window inference on a large satellite image.

        Args:
            image (np.ndarray): (H, W, C) full image.
            patch_size (int): Patch side length.
            stride (int): Sliding window stride.
            stats (dict | None): Normalisation stats {'mean', 'std'}.

        Returns:
            dict: 'prediction' (H, W), 'confidence' (H, W), 'probabilities' (H, W, C).
        """
        h, w, c = image.shape
        prob_map = np.zeros((h, w, self.num_classes), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for row in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                patch = image[row:row + patch_size, col:col + patch_size, :].copy()

                if stats is not None:
                    mean = stats["mean"].reshape(1, 1, -1)
                    std = stats["std"].reshape(1, 1, -1)
                    patch = (patch - mean) / (std + 1e-8)

                result = self.predict_patch(patch)
                prob_map[row:row + patch_size, col:col + patch_size, :] += result["probabilities"]
                count_map[row:row + patch_size, col:col + patch_size] += 1

        # Avoid division by zero
        count_map = np.maximum(count_map, 1)
        prob_map /= count_map[..., np.newaxis]

        prediction = prob_map.argmax(axis=-1).astype(np.int32)
        confidence = prob_map.max(axis=-1)

        logger.info("Large-image inference complete: %dx%d", h, w)
        return {
            "prediction": prediction,
            "probabilities": prob_map,
            "confidence": confidence,
        }

    @staticmethod
    def load_checkpoint(model, checkpoint_path, device="cpu"):
        """
        Load model weights from a checkpoint file.

        Args:
            model (nn.Module): Model to load weights into.
            checkpoint_path (str): Path to .pth checkpoint.
            device (str): Device to map weights to.

        Returns:
            nn.Module: Model with loaded weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                "Loaded checkpoint: epoch=%d, metric=%.4f",
                checkpoint.get("epoch", -1),
                checkpoint.get("metric", float("nan")),
            )
        else:
            model.load_state_dict(checkpoint)
        return model
