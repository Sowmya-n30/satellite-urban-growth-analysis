"""
ResNet50-based models for satellite image classification and segmentation.

Provides:
    - ResNet50Classifier: Image-level classification (urban / semi / non-urban)
    - ResNet50Segmenter: Feature-Pyramid-style segmentation head on ResNet50 backbone

Supports pre-trained ImageNet weights with fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class ResNet50Classifier(nn.Module):
    """
    ResNet50 fine-tuned for satellite patch classification.

    The first convolution is replaced to accept an arbitrary number of
    input channels (Sentinel-2 has 10 bands, not 3).

    Args:
        in_channels (int): Number of input spectral bands (default 10).
        num_classes (int): Number of output classes (default 3).
        pretrained (bool): Load ImageNet weights (default True).
        freeze_backbone (bool): Freeze backbone during early training.
        dropout (float): Dropout before the classification head.
    """

    def __init__(
        self,
        in_channels=10,
        num_classes=3,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.3,
    ):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required. Install: pip install torchvision")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Replace the stem conv for multi-spectral input
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        # Initialize new conv; average ImageNet weights over the channel dim if pretrained
        if pretrained and in_channels != 3:
            with torch.no_grad():
                # Tile / average the 3-channel weights to in_channels
                w = original_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                backbone.conv1.weight.copy_(w.repeat(1, in_channels, 1, 1))

        # Remove the original FC head; keep feature extractor
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)           # (B, 2048, H/32, W/32)
        pooled = self.pool(features).flatten(1)  # (B, 2048)
        return self.classifier(pooled)

    def unfreeze_backbone(self):
        """Un-freeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNet50Segmenter(nn.Module):
    """
    ResNet50 encoder with a simple decoder head for semantic segmentation.

    Uses multi-scale feature maps from ResNet50 layers to produce
    pixel-wise predictions.

    Args:
        in_channels (int): Input spectral bands.
        num_classes (int): Output classes.
        pretrained (bool): Use ImageNet weights.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels=10, num_classes=3, pretrained=True, dropout=0.2):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required.")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Adapt first conv
        original_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained and in_channels != 3:
            with torch.no_grad():
                w = original_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1))
        elif pretrained:
            new_conv.weight = original_conv.weight

        self.layer0 = nn.Sequential(new_conv, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch, /4
        self.layer2 = backbone.layer2   # 512 ch, /8
        self.layer3 = backbone.layer3   # 1024 ch, /16
        self.layer4 = backbone.layer4   # 2048 ch, /32

        # Decoder
        self.dec4 = self._decoder_block(2048, 512, dropout)
        self.dec3 = self._decoder_block(512 + 1024, 256, dropout)
        self.dec2 = self._decoder_block(256 + 512, 128, dropout)
        self.dec1 = self._decoder_block(128 + 256, 64, dropout)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _decoder_block(in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        x0 = self.layer0(x)    # /4
        x1 = self.layer1(x0)   # /4
        x2 = self.layer2(x1)   # /8
        x3 = self.layer3(x2)   # /16
        x4 = self.layer4(x3)   # /32

        # Decoder with skip connections
        d4 = self.dec4(x4)
        d4_up = F.interpolate(d4, size=x3.shape[2:], mode="bilinear", align_corners=False)

        d3 = self.dec3(torch.cat([d4_up, x3], dim=1))
        d3_up = F.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=False)

        d2 = self.dec2(torch.cat([d3_up, x2], dim=1))
        d2_up = F.interpolate(d2, size=x1.shape[2:], mode="bilinear", align_corners=False)

        d1 = self.dec1(torch.cat([d2_up, x1], dim=1))
        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out


def build_resnet50_classifier(in_channels=10, num_classes=3, pretrained=True):
    """Factory for ResNet50Classifier."""
    model = ResNet50Classifier(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet50Classifier | Total params: {total:,}")
    return model


def build_resnet50_segmenter(in_channels=10, num_classes=3, pretrained=True):
    """Factory for ResNet50Segmenter."""
    model = ResNet50Segmenter(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet50Segmenter | Total params: {total:,}")
    return model
