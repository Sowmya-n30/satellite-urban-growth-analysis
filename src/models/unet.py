"""
U-Net for semantic segmentation of multi-spectral satellite imagery.

Architecture:
    - Encoder: 4 downsampling blocks with double-convolution
    - Bottleneck: Double-convolution at lowest resolution
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Per-pixel class probabilities

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation," MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BatchNorm → ReLU) blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool followed by DoubleConv (encoder step)."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample (bilinear or transposed conv) + DoubleConv (decoder step)."""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1 convolution to produce final class-score maps."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation of satellite imagery.

    Args:
        in_channels (int): Number of input spectral bands (default 10).
        num_classes (int): Number of output classes (default 3).
        base_features (int): Feature channels in the first encoder block.
        bilinear (bool): Use bilinear upsampling (True) or transposed conv (False).
        dropout (float): Dropout rate in DoubleConv blocks.
    """

    def __init__(
        self,
        in_channels=10,
        num_classes=3,
        base_features=64,
        bilinear=True,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        f = base_features
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, f, dropout=dropout)
        self.down1 = Down(f, f * 2, dropout=dropout)
        self.down2 = Down(f * 2, f * 4, dropout=dropout)
        self.down3 = Down(f * 4, f * 8, dropout=dropout)
        self.down4 = Down(f * 8, f * 16 // factor, dropout=dropout)

        self.up1 = Up(f * 16, f * 8 // factor, bilinear, dropout=dropout)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear, dropout=dropout)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear, dropout=dropout)
        self.up4 = Up(f * 2, f, bilinear, dropout=dropout)

        self.outc = OutConv(f, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def get_num_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_unet(in_channels=10, num_classes=3, base_features=64, bilinear=True, dropout=0.1):
    """Convenience factory function for UNet."""
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_features=base_features,
        bilinear=bilinear,
        dropout=dropout,
    )
    total, trainable = model.get_num_parameters()
    print(f"UNet | Total params: {total:,} | Trainable: {trainable:,}")
    return model
