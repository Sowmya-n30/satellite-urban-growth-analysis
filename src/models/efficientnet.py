"""
EfficientNet-based models for satellite image classification and segmentation.

Provides:
    - EfficientNetClassifier: Lightweight image-level classification
    - EfficientNetSegmenter: EfficientNet encoder with lightweight decoder

Supports EfficientNet-B0 through B7 via torchvision (>= 0.13).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import (
        efficientnet_b0, EfficientNet_B0_Weights,
        efficientnet_b3, EfficientNet_B3_Weights,
        efficientnet_b4, EfficientNet_B4_Weights,
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


VARIANT_MAP = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights, 1280),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights, 1536),
    "b4": (efficientnet_b4, EfficientNet_B4_Weights, 1792),
}


def _patch_first_conv(model, in_channels, pretrained):
    """Replace the first Conv2d to accept in_channels (not 3)."""
    # EfficientNet stem: model.features[0][0] is a Conv2dNormActivation -> [0] is Conv2d
    stem_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        bias=False,
    )
    if pretrained and in_channels != 3:
        with torch.no_grad():
            w = stem_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1))
    elif pretrained:
        new_conv.weight = stem_conv.weight
    model.features[0][0] = new_conv
    return model


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based patch classifier for satellite imagery.

    Args:
        in_channels (int): Number of input spectral bands (default 10).
        num_classes (int): Number of output classes.
        variant (str): 'b0', 'b3', or 'b4'.
        pretrained (bool): Use ImageNet weights.
        dropout (float): Dropout before the final linear layer.
    """

    def __init__(
        self,
        in_channels=10,
        num_classes=3,
        variant="b0",
        pretrained=True,
        dropout=0.2,
    ):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision >= 0.13 required. Install: pip install torchvision")
        if variant not in VARIANT_MAP:
            raise ValueError(f"variant must be one of {list(VARIANT_MAP.keys())}")

        model_fn, weights_cls, out_features = VARIANT_MAP[variant]
        weights = weights_cls.DEFAULT if pretrained else None
        backbone = model_fn(weights=weights)
        backbone = _patch_first_conv(backbone, in_channels, pretrained)

        # Remove original classifier head; keep feature extraction
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EfficientNetSegmenter(nn.Module):
    """
    EfficientNet-B0 encoder with a lightweight FPN-style decoder.

    Extracts multi-scale features at strides 4, 8, 16, 32 and
    progressively upsamples to the original resolution.

    Args:
        in_channels (int): Input spectral bands.
        num_classes (int): Output classes.
        pretrained (bool): ImageNet weights.
        dropout (float): Dropout in decoder.
    """

    def __init__(self, in_channels=10, num_classes=3, pretrained=True, dropout=0.2):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision >= 0.13 required.")

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        backbone = _patch_first_conv(backbone, in_channels, pretrained)

        # Extract multi-scale feature blocks from EfficientNet-B0
        self.enc0 = backbone.features[0]        # stride 2, 32 ch
        self.enc1 = backbone.features[1]        # stride 2, 16 ch
        self.enc2 = nn.Sequential(*backbone.features[2:4])
        self.enc3 = nn.Sequential(*backbone.features[4:6])
        self.enc4 = nn.Sequential(*backbone.features[6:])

        # Decoder
        self.dec3 = self._block(1280 + 48, 128, dropout)
        self.dec2 = self._block(128 + 24, 64, dropout)
        self.dec1 = self._block(64 + 16, 32, dropout)
        self.dec0 = self._block(32 + 32, 16, dropout)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    @staticmethod
    def _block(in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        input_size = x.shape[2:]

        e0 = self.enc0(x)          # (B, 32, H/2, W/2)
        e1 = self.enc1(e0)         # (B, 16, H/2, W/2)
        e2 = self.enc2(e1)         # (B, 24, H/4, W/4)
        e3 = self.enc3(e2)         # (B, 48, H/8, W/8)
        e4 = self.enc4(e3)         # (B, 1280, H/32, W/32)

        d3 = self.dec3(torch.cat([
            F.interpolate(e4, size=e3.shape[2:], mode="bilinear", align_corners=False), e3
        ], dim=1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1
        ], dim=1))
        d0 = self.dec0(torch.cat([
            F.interpolate(d1, size=e0.shape[2:], mode="bilinear", align_corners=False), e0
        ], dim=1))

        out = self.final(d0)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out


def build_efficientnet_classifier(in_channels=10, num_classes=3, variant="b0", pretrained=True):
    """Factory for EfficientNetClassifier."""
    model = EfficientNetClassifier(
        in_channels=in_channels, num_classes=num_classes,
        variant=variant, pretrained=pretrained,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-{variant} Classifier | Total params: {total:,}")
    return model


def build_efficientnet_segmenter(in_channels=10, num_classes=3, pretrained=True):
    """Factory for EfficientNetSegmenter."""
    model = EfficientNetSegmenter(
        in_channels=in_channels, num_classes=num_classes, pretrained=pretrained
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"EfficientNetSegmenter | Total params: {total:,}")
    return model
