import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights

class _ConvRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=True),
            nn.ReLU(inplace=True),
        )


def load_vgg16_frontend(csrnet, freeze_frontend: bool = False):
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
    mapping = [
        (0,  csrnet.conv1_1[0]),
        (2,  csrnet.conv1_2[0]),
        (5,  csrnet.conv2_1[0]),
        (7,  csrnet.conv2_2[0]),
        (10, csrnet.conv3_1[0]),
        (12, csrnet.conv3_2[0]),
        (14, csrnet.conv3_3[0]),
        (17, csrnet.conv4_1[0]),
        (19, csrnet.conv4_2[0]),
        (21, csrnet.conv4_3[0]),
    ]
    for vgg_idx, csr_conv in mapping:
        vgg_conv = vgg[vgg_idx]
        assert isinstance(vgg_conv, torch.nn.Conv2d)
        assert isinstance(csr_conv, torch.nn.Conv2d)
        csr_conv.weight.data.copy_(vgg_conv.weight.data)
        csr_conv.bias.data.copy_(vgg_conv.bias.data)
    if freeze_frontend:
        for p in [
            *csrnet.conv1_1.parameters(), *csrnet.conv1_2.parameters(),
            *csrnet.conv2_1.parameters(), *csrnet.conv2_2.parameters(),
            *csrnet.conv3_1.parameters(), *csrnet.conv3_2.parameters(), *csrnet.conv3_3.parameters(),
            *csrnet.conv4_1.parameters(), *csrnet.conv4_2.parameters(), *csrnet.conv4_3.parameters(),
        ]:
            p.requires_grad = False

class CSRNet(nn.Module):
    """
    CSRNet-style fully convolutional network for crowd counting / density estimation.

    Rough architecture (following the paper, simplified):
    - Front-end: VGG-like conv blocks with pooling (downsample by 8x).
    - Back-end: dilated convolutions to enlarge receptive field without further downsampling.
    - Output: single-channel density map, optionally passed through an activation.
    """

    def __init__(self, *, output_activation: str = "none"):
        super().__init__()
        if output_activation not in ("relu", "softplus", "none"):
            raise ValueError("output_activation must be 'relu', 'softplus', or 'none'")
        self._output_activation = output_activation

        # ---- Front-end (VGG-style) ----
        # conv1
        self.conv1_1 = _ConvRelu(3, 64)
        self.conv1_2 = _ConvRelu(64, 64)
        self.pool1 = nn.MaxPool2d(2)  # 1/2

        # conv2
        self.conv2_1 = _ConvRelu(64, 128)
        self.conv2_2 = _ConvRelu(128, 128)
        self.pool2 = nn.MaxPool2d(2)  # 1/4

        # conv3
        self.conv3_1 = _ConvRelu(128, 256)
        self.conv3_2 = _ConvRelu(256, 256)
        self.conv3_3 = _ConvRelu(256, 256)
        self.pool3 = nn.MaxPool2d(2)  # 1/8

        # conv4 (no further pooling, as in CSRNet)
        self.conv4_1 = _ConvRelu(256, 512)
        self.conv4_2 = _ConvRelu(512, 512)
        self.conv4_3 = _ConvRelu(512, 512)

        # ---- Back-end (dilated convs) ----
        self.dilated5_1 = _ConvRelu(512, 512, d=2, p=2)
        self.dilated5_2 = _ConvRelu(512, 512, d=2, p=2)
        self.dilated5_3 = _ConvRelu(512, 512, d=2, p=2)

        self.dilated6_1 = _ConvRelu(512, 256, d=2, p=2)
        self.dilated6_2 = _ConvRelu(256, 128, d=2, p=2)
        self.dilated6_3 = _ConvRelu(128, 64, d=2, p=2)

        # final 1x1 conv to regress density
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # front-end
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)

        # back-end (dilated)
        x = self.dilated5_1(x)
        x = self.dilated5_2(x)
        x = self.dilated5_3(x)

        x = self.dilated6_1(x)
        x = self.dilated6_2(x)
        x = self.dilated6_3(x)

        density = self.out(x)

        if self._output_activation == "relu":
            density = F.relu(density)
        elif self._output_activation == "softplus":
            density = F.softplus(density)

        return density

