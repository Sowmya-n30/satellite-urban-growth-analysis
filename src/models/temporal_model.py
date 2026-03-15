"""
Temporal change-detection model using a Siamese network architecture.

The model compares two co-registered satellite image patches (t1 and t2)
and outputs a per-pixel change-probability map.

Architecture:
    - Shared CNN encoder (Siamese branches)
    - Feature difference / concatenation
    - Decoder producing binary or multi-class change maps

Optional LSTM fusion for multi-step temporal sequences (>2 time steps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import DoubleConv, Down, Up, OutConv


class SiameseEncoder(nn.Module):
    """Shared encoder used in both Siamese branches."""

    def __init__(self, in_channels, base_features=64, dropout=0.1):
        super().__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f, dropout=dropout)
        self.down1 = Down(f, f * 2, dropout=dropout)
        self.down2 = Down(f * 2, f * 4, dropout=dropout)
        self.down3 = Down(f * 4, f * 8, dropout=dropout)
        self.down4 = Down(f * 8, f * 16, dropout=dropout)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class SiameseChangeDetector(nn.Module):
    """
    Siamese U-Net for binary change detection between two time points.

    Takes two images (t1, t2) sharing the same encoder weights and
    produces a pixel-wise change probability map.

    Args:
        in_channels (int): Spectral bands per image (default 10).
        num_classes (int): 2 for binary change, more for multi-class.
        base_features (int): Encoder base feature depth.
        bilinear (bool): Bilinear upsampling in decoder.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels=10,
        num_classes=2,
        base_features=64,
        bilinear=True,
        dropout=0.1,
    ):
        super().__init__()
        f = base_features

        # Shared encoder (outputs: f, f*2, f*4, f*8, f*16 channels)
        self.encoder = SiameseEncoder(in_channels, base_features, dropout)

        # Bottleneck: fuse t1 and t2 bottleneck features
        self.bottleneck_conv = DoubleConv(f * 16 * 2, f * 16, dropout=dropout)

        # Decoder: in_channels = upsampled + skip connection channels
        self.up1 = Up(f * 16 + f * 8, f * 8, bilinear, dropout=dropout)
        self.up2 = Up(f * 8 + f * 4, f * 4, bilinear, dropout=dropout)
        self.up3 = Up(f * 4 + f * 2, f * 2, bilinear, dropout=dropout)
        self.up4 = Up(f * 2 + f, f, bilinear, dropout=dropout)
        self.outc = OutConv(f, num_classes)

    def forward(self, t1, t2):
        """
        Args:
            t1 (Tensor): Earlier-date image (B, C, H, W).
            t2 (Tensor): Later-date image (B, C, H, W).

        Returns:
            Tensor: Change logits (B, num_classes, H, W).
        """
        # Encode both images with shared weights
        s1_1, s1_2, s1_3, s1_4, s1_5 = self.encoder(t1)
        s2_1, s2_2, s2_3, s2_4, s2_5 = self.encoder(t2)

        # Fuse bottleneck features
        fused = self.bottleneck_conv(torch.cat([s1_5, s2_5], dim=1))

        # Decode with skip connections from t2 encoder
        x = self.up1(fused, s2_4)
        x = self.up2(x, s2_3)
        x = self.up3(x, s2_2)
        x = self.up4(x, s2_1)

        return self.outc(x)


class TemporalLSTMModel(nn.Module):
    """
    Multi-temporal change detection model with ConvLSTM-style fusion.

    Encodes a sequence of T images and produces a change-detection map
    by comparing the hidden state at each timestep.

    Args:
        in_channels (int): Spectral bands per image.
        num_classes (int): Output classes.
        hidden_dim (int): LSTM hidden dimension.
        num_time_steps (int): Number of time steps T.
        base_features (int): CNN feature base depth.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels=10,
        num_classes=2,
        hidden_dim=256,
        num_time_steps=4,
        base_features=64,
        dropout=0.1,
    ):
        super().__init__()
        self.num_time_steps = num_time_steps
        f = base_features

        # CNN encoder (shared across time steps)
        self.cnn_encoder = nn.Sequential(
            DoubleConv(in_channels, f, dropout=dropout),
            Down(f, f * 2, dropout=dropout),
            Down(f * 2, f * 4, dropout=dropout),
            Down(f * 4, f * 8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        cnn_out_dim = f * 8 * 8 * 8
        num_layers = 2
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, images):
        """
        Args:
            images (Tensor): (B, T, C, H, W) – sequence of T images.

        Returns:
            Tensor: (B, num_classes) – per-sample class logits.
        """
        B, T, C, H, W = images.shape
        features = []
        for t in range(T):
            feat = self.cnn_encoder(images[:, t])  # (B, f*8, H', W')
            feat = self.pool(feat)                  # (B, f*8, 8, 8)
            features.append(feat.flatten(1))        # (B, f*8*64)
        seq = torch.stack(features, dim=1)          # (B, T, feature_dim)

        lstm_out, _ = self.lstm(seq)                # (B, T, hidden_dim)
        last_hidden = lstm_out[:, -1, :]            # (B, hidden_dim)
        return self.decoder(last_hidden)


def build_siamese_model(in_channels=10, num_classes=2, base_features=64):
    """Factory for SiameseChangeDetector."""
    model = SiameseChangeDetector(
        in_channels=in_channels, num_classes=num_classes, base_features=base_features
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"SiameseChangeDetector | Total params: {total:,}")
    return model


def build_temporal_model(in_channels=10, num_classes=2, num_time_steps=4):
    """Factory for TemporalLSTMModel."""
    model = TemporalLSTMModel(
        in_channels=in_channels,
        num_classes=num_classes,
        num_time_steps=num_time_steps,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"TemporalLSTMModel | Total params: {total:,}")
    return model
