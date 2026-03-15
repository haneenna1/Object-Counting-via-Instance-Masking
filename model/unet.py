import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Original U-Net block: two 3x3 convolutions, each followed by ReLU. No BatchNorm."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDensity(nn.Module):
    """
    U-Net as in the original paper (Ronneberger et al., 2015):
    Contracting path: 64 → 128 → 256 → 512 → 1024 (bottleneck).
    Symmetric expanding path with skip connections by concatenation.
    Two 3×3 convs + ReLU per block, no BatchNorm.
    """

    def __init__(self, *, output_activation: str = "softplus"):
        super().__init__()
        if output_activation not in ("relu", "softplus", "none"):
            raise ValueError("output_activation must be 'relu', 'softplus', or 'none'")
        self._output_activation = output_activation

        # Contracting path (encoder): 64 → 128 → 256 → 512 → 1024
        self.enc1 = ConvBlock(3, 64)   # 64
        self.enc2 = ConvBlock(64, 128)  # 128
        self.enc3 = ConvBlock(128, 256) # 256
        self.enc4 = ConvBlock(256, 512) # 512
        self.bottleneck = ConvBlock(512, 1024)  # 1024

        self.pool = nn.MaxPool2d(2)

        # Expanding path (decoder): up-conv then concat then two convs
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)  # 512 + 512 from skip

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)   # 256 + 256 from skip

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)   # 128 + 128 from skip

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)    # 64 + 64 from skip

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Contracting path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Expanding path
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        density = self.out(d1)
        if self._output_activation == "relu":
            density = F.relu(density)
        elif self._output_activation == "softplus":
            density = F.softplus(density)

        return density
