import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ViTDensity(nn.Module):
    """
    Pretrained ViT encoder (via timm) + lightweight convolutional decoder
    for density-map regression.

    The ViT produces patch embeddings at 1/patch_size resolution (typically 1/16).
    The decoder upsamples back to full input resolution with 4 stages of 2x
    transposed convolutions, producing a single-channel density map.

    Input must be divisible by patch_size in both spatial dimensions.
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        output_activation: str = "none",
    ):
        super().__init__()
        if output_activation not in ("relu", "softplus", "none"):
            raise ValueError("output_activation must be 'relu', 'softplus', or 'none'")
        self._output_activation = output_activation

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            # dynamic_img_size=True,
        )
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.embed_dim = self.encoder.embed_dim
        self.num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", 1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 4 stages of 2x upsampling: 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
        self.decoder = nn.Sequential(
            _DecoderBlock(self.embed_dim, 256),
            _DecoderBlock(256, 128),
            _DecoderBlock(128, 64),
            _DecoderBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        p = self.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            print(f"warning: padding input image to be divisible by {p}")
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        H_pad, W_pad = x.shape[-2:]

        features = self.encoder.forward_features(x)
        features = features[:, self.num_prefix_tokens:, :]

        h = H_pad // p
        w = W_pad // p
        features = features.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        density = self.decoder(features)
        density = density[:, :, :H, :W]

        if self._output_activation == "relu":
            density = F.relu(density)
        elif self._output_activation == "softplus":
            density = F.softplus(density)

        return density
