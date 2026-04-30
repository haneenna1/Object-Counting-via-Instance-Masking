"""
Minimal training script for density map regression (object counting)
with logging and training curves.

Dataset yields raw density (sum ≈ count).

Two GT supervision modes when pred is lower-res than GT (e.g. CSRNet 1/8):

- ``bilinear`` (default for non-CSRNet): MSE on **raw** pred vs **raw** (resized, area-corrected)
  GT; ``density_loss = density_scale * MSE(pred, gt)`` so gradients are scaled up. Count:
  ``pred.sum()`` (same units as GT).

- ``csrnet_cubic`` (CSRNet-pytorch ``image.py``): ``cv2.resize(..., INTER_CUBIC)``
  to pred size, then ``* 64`` to preserve count. Use ``density_scale=1``.
  Count: ``pred.sum()``.
"""

import json
import random
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm

from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR

from data.dataset import PatchAugmentedDataset
from data.dataset import visualize_image_and_density

import os
# Default training scale (only applied in this module, not in dataset).
DEFAULT_DENSITY_SCALE = 255.0

# CSRNet reference: downscale GT by 8×8 → multiply by 64 to keep integral (see CSRNet-pytorch image.py).
CSRNET_GT_DOWNSAMPLE_FACTOR_SQ = 64.0

# ImageNet normalization stats (for B2 mask fill).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

MASK_FILL_MODES = ("imagenet_mean", "zero", "noise", "learnable")


def apply_mask_fill(
    images: torch.Tensor,
    mask: torch.Tensor | None,
    mode: str,
    mask_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Override the dataset's default zero-fill of the masked region.

    The dataset multiplies the normalized image by ``(1 - mask)``, which leaves
    masked pixels at 0 in normalized space -- i.e. ImageNet mean in pixel space.
    This helper adds a per-channel offset inside the mask region so the
    effective fill becomes one of:

    - ``imagenet_mean`` (no-op; default and backward-compatible).
    - ``zero``: literal pixel 0 (black). In normalized space = ``-mean/std``.
    - ``noise``: per-pixel N(0, 1) noise in normalized space.
    - ``learnable``: a per-channel learnable parameter (MAE-style mask token).
      ``mask_token`` must be a ``(3,)`` parameter trained jointly with the model.

    ``images`` is expected to be the normalized, dataset-produced tensor
    ``(B, 3, H, W)``. ``mask`` may be ``None`` or have any spatial size; it
    is resized with nearest-neighbor to match ``images`` if needed. When the
    mask is all zero the tensor is returned unchanged.
    """
    if mode not in MASK_FILL_MODES:
        raise ValueError(f"mask_fill mode must be in {MASK_FILL_MODES}, got {mode!r}")
    if mode == "imagenet_mean" or mask is None:
        return images
    if mask.shape[-2:] != images.shape[-2:]:
        m = F.interpolate(mask, size=images.shape[-2:], mode="nearest")
    else:
        m = mask
    m = m.clamp(0.0, 1.0)
    if float(m.sum().item()) == 0.0:
        return images

    if mode == "zero":
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        fill = -mean / std
        return images + fill * m
    if mode == "noise":
        fill = torch.randn_like(images)
        return images + fill * m
    if mode == "learnable":
        if mask_token is None:
            raise ValueError("mask_fill='learnable' requires a mask_token parameter.")
        if mask_token.ndim != 1 or mask_token.shape[0] != 3:
            raise ValueError(
                f"mask_token must have shape (3,), got {tuple(mask_token.shape)}."
            )
        fill = mask_token.to(images.dtype).view(1, 3, 1, 1)
        return images + fill * m
    raise ValueError(f"Unknown mask_fill mode {mode!r}")


def _param_skips_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    """ViT / transformer-style: no decay on bias, norm, and token/position embeddings."""
    if param.ndim <= 1:
        return True
    n = name.lower()
    if any(
        t in n
        for t in (
            "pos_embed",
            "cls_token",
            "dist_token",
            "mask_token",
            "register_tokens",
            "prompt",
            "rel_pos",
            "relative_position_bias_table",
            "absolute_pos_embed",
        )
    ):
        return True
    if "norm" in n:
        return True
    return False


def build_optimizer_param_groups(
    module: torch.nn.Module,
    dec_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Build decoder/encoder decay groups, with encoder LR initialized to 0."""
    if not hasattr(module, "encoder"):
        raise RuntimeError("Model has no `encoder` module.")
    if not hasattr(module, "decoder"):
        raise RuntimeError("Model has no `decoder` module.")

    dec_decay: list[torch.nn.Parameter] = []
    dec_no_decay: list[torch.nn.Parameter] = []
    enc_decay: list[torch.nn.Parameter] = []
    enc_no_decay: list[torch.nn.Parameter] = []

    for name, param in module.decoder.named_parameters():
        if not param.requires_grad:
            continue
        if _param_skips_weight_decay(name, param):
            dec_no_decay.append(param)
        else:
            dec_decay.append(param)

    for name, param in module.encoder.named_parameters():
        if _param_skips_weight_decay(name, param):
            enc_no_decay.append(param)
        else:
            enc_decay.append(param)

    groups: list[dict] = []
    if dec_no_decay:
        groups.append(
            {"params": dec_no_decay, "lr": dec_lr, "weight_decay": 0.0, "group_name": "decoder_no_decay"}
        )
    if dec_decay:
        groups.append(
            {"params": dec_decay, "lr": dec_lr, "weight_decay": weight_decay, "group_name": "decoder_decay"}
        )
    if enc_no_decay:
        groups.append(
            {"params": enc_no_decay, "lr": backbone_lr, "weight_decay": 0.0, "group_name": "encoder_no_decay"}
        )
    if enc_decay:
        groups.append(
            {"params": enc_decay, "lr": backbone_lr, "weight_decay": 0.05, "group_name": "encoder_decay"}
        )

    if hasattr(module, "mask_token") and module.mask_token.requires_grad:
        groups.append(
            {
                "params": [module.mask_token],
                "lr": dec_lr,
                "weight_decay": 0.0,
                "group_name": "mask_token",
            }
        )

    # Encoder-only inpainting auxiliary head (1 linear layer, see ViTDensity).
    # Treated as a small head: decoder-side learning rate, but with weight decay
    # because it's a non-bias matmul. Only added when the model was constructed
    # with hidden_count_aux=True; otherwise the attribute is None.
    aux_head = getattr(module, "aux_head", None)
    if aux_head is not None:
        aux_decay: list[torch.nn.Parameter] = []
        aux_no_decay: list[torch.nn.Parameter] = []
        for name, param in aux_head.named_parameters():
            if not param.requires_grad:
                continue
            if _param_skips_weight_decay(name, param):
                aux_no_decay.append(param)
            else:
                aux_decay.append(param)
        if aux_no_decay:
            groups.append(
                {"params": aux_no_decay, "lr": dec_lr, "weight_decay": 0.0, "group_name": "aux_head_no_decay"}
            )
        if aux_decay:
            groups.append(
                {"params": aux_decay, "lr": dec_lr, "weight_decay": weight_decay, "group_name": "aux_head_decay"}
            )

    if not groups:
        raise RuntimeError("No trainable parameters in module for optimizer groups.")
    return groups


def downsample_gt_csrnet_cubic(
    gt_density: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """
    Match CSRNet-pytorch: cv2.resize to (out_w, out_h), INTER_CUBIC, × 64.

    ``gt_density``: (N, 1, H, W) or (N, C, H, W) with C==1.
    """
    if gt_density.shape[1] != 1:
        raise ValueError("csrnet_cubic downsample expects a single-channel density (N,1,H,W).")
    device = gt_density.device
    dtype = gt_density.dtype
    x = gt_density.detach().float().cpu().numpy()
    n = x.shape[0]
    out = np.empty((n, 1, out_h, out_w), dtype=np.float32)
    for i in range(n):
        plane = x[i, 0]
        resized = cv2.resize(plane, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        out[i, 0] = resized.astype(np.float32) * CSRNET_GT_DOWNSAMPLE_FACTOR_SQ
    return torch.from_numpy(out).to(device=device, dtype=dtype)

class PatchBatchSampler(BatchSampler):
    """
    Yield one full patch group per batch for PatchAugmentedDataset.

    Each batch contains all variants generated from a single base image,
    which keeps spatial shapes consistent inside the batch.
    """

    def __init__(self, dataset: PatchAugmentedDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        base_indices = list(range(len(self.dataset.dataset)))
        if self.shuffle:
            random.shuffle(base_indices)

        group_size = self.dataset.variants_per_image
        for base_idx in base_indices:
            start = base_idx * group_size
            yield list(range(start, start + group_size))

    def __len__(self) -> int:
        return len(self.dataset.dataset)


class CountWeightController:
    """
    w(t) = w_max * (1 - mse_ema / mse_ref) ** p, clamped to [0, w_max].

    - mse_ref: a reference MSE (e.g. EMA value at end of warmup).
    - p > 1 delays the ramp until MSE is well below mse_ref.
    - p < 1 ramps faster.
    """
    def __init__(self, w_max=0.01, p=2.0, ema=0.9, warmup_epochs=5):
        self.w_max = w_max
        self.p = p
        self.ema = ema
        self.warmup = warmup_epochs
        self.mse_ema = None
        self.mse_ref = None

    def update(self, epoch, train_mse):
        if self.mse_ema is None:
            self.mse_ema = train_mse
        else:
            self.mse_ema = self.ema * self.mse_ema + (1 - self.ema) * train_mse

        if epoch == self.warmup:
            self.mse_ref = self.mse_ema  # freeze reference after warmup

        if self.mse_ref is None or epoch < self.warmup:
            return 0.0
        progress = max(0.0, 1.0 - self.mse_ema / max(self.mse_ref, 1e-8))
        return self.w_max * (progress ** self.p)

# -----------------------------
# Loss: density MSE + choice of count loss (MAE or MSE on count)
# -----------------------------
def _gt_density_resize(gt_density, pred_density, gt_downsample, density_scale):
    if gt_downsample not in ("bilinear", "csrnet_cubic"):
        raise ValueError(
            f"gt_downsample must be 'bilinear' or 'csrnet_cubic', got {gt_downsample!r}"
        )
    H_pred, W_pred = pred_density.shape[-2:]

    if pred_density.shape[-2:] != gt_density.shape[-2:]:
        if gt_downsample == "csrnet_cubic":
            gt_resized = downsample_gt_csrnet_cubic(gt_density, H_pred, W_pred)
            gt_for_count = gt_resized
            gt_target = gt_resized
        else:
            H_gt, W_gt = gt_density.shape[-2:]
            gt_resized = F.interpolate(
                gt_density,
                size=(H_pred, W_pred),
                mode="bilinear",
                align_corners=False,
            )
            spatial_scale = (H_gt / H_pred) * (W_gt / W_pred)
            gt_resized = gt_resized * spatial_scale
            gt_for_count = gt_resized
            gt_target = gt_resized * density_scale
    else:
        gt_for_count = gt_density
        gt_target = gt_density * density_scale

    return gt_for_count, gt_target

def compute_loss(
    pred_density,
    gt_density,
    count_loss_weight: float = 1.0,
    invisible_density_weight: float = 1.0,
    invisible_count_weight: float = 1.0,
    loss_mode: str = "density_mse_count_l1",
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    mask: torch.Tensor | None = None,
    mask_mode: str = "robust",  # retained for API compat; no longer branched on
    invisible_density_norm: str = "region_mean",
    eps: float = 1e-6,
):
    """
    Unified density + count loss with visible / invisible supervision.
    Density term
        Two normalizations controlled by ``invisible_density_norm``:
        - ``region_mean`` (default): per-pixel MSE inside each region,
          normalized by that region's pixel count (area-invariant):
              L_dens = mean_batch( vis_mse_px + w_inv_den * inv_mse_px )
          Makes visible and invisible numerically comparable but
          under-supervises the (usually tiny) invisible region because its
          gradient magnitude is independent of how much you asked the model
          to hallucinate.
        - ``area_scaled``: both regions are normalized by the same total
          pixel count, so the invisible term's magnitude scales with the
          mask area fraction:
              L_dens = mean_batch( (vis_err + w * inv_err) / total_pix )
          This bumps the gradient on the hallucination task as soon as
          the mask is non-trivial -- closer to "sum of squared errors"
          with mask-weighting.
    Count term (pixel-normalization bug fixed)
        Counts are scalars in units of "heads", not "heads per pixel", so we
        do NOT divide by region pixel count. We use relative (per-head)
        normalization with clamp_min(1.0):
            |dC_r| / max(GT_count_r, 1)
        - In inpaint mode (GT_inv > 0) this is the relative miscount rate
          for hidden heads -- commensurable with the visible rate.
        - In robust mode (GT_inv == 0 by construction, because the dataset
          zeros GT density inside the mask) this collapses safely to
          |Σpred_inv|, which correctly penalizes any predicted density
          inside the masked region (pushes pred -> 0 there).
    No-mask case (``mask is None`` or all-zero)
        Invisible region has zero area, both invisible terms collapse to
        zero, and the loss reduces to visible density MSE + visible count
        L1/L2 -- behaviorally equivalent to the previous no-mask branch.
    """

    # -----------------------------
    # Resize GT density (preserve count)
    # -----------------------------
    gt_for_count, gt_target = _gt_density_resize(gt_density, pred_density, gt_downsample, density_scale)

    if mask is None:
        mask = torch.zeros_like(pred_density)
    else:
        if mask.shape[-2:] != pred_density.shape[-2:]:
            mask = F.interpolate(mask, size=pred_density.shape[-2:], mode="nearest")
        mask = mask.clamp(0.0, 1.0)
    vis_mask = 1.0 - mask

    # -----------------------------
    # Density loss (region-normalized by pixels; correct for summed MSE)
    # -----------------------------
    sq_err = (pred_density - gt_target) ** 2

    vis_err = (sq_err * vis_mask).sum(dim=(1, 2, 3))
    inv_err = (sq_err * mask).sum(dim=(1, 2, 3))

    vis_pix = vis_mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    inv_pix = mask.sum(dim=(1, 2, 3)).clamp_min(eps)

    if invisible_density_norm not in ("region_mean", "area_scaled"):
        raise ValueError(
            f"invisible_density_norm must be 'region_mean' or 'area_scaled', "
            f"got {invisible_density_norm!r}"
        )
    if invisible_density_norm == "region_mean":
        # When a region is empty (e.g. no mask), its sq_err contribution is also 0,
        # so the /eps guard is harmless (0 / eps == 0).
        density_loss = (
            (vis_err / vis_pix)
            + float(invisible_density_weight) * (inv_err / inv_pix)
        ).mean()
    else:
        total_pix = (vis_pix + inv_pix).clamp_min(eps)
        density_loss = (
            (vis_err + float(invisible_density_weight) * inv_err) / total_pix
        ).mean()

    # -----------------------------
    # Count loss (per-region; NO pixel normalization)
    # -----------------------------
    pred_raw = pred_density / density_scale

    pred_visible_count = (pred_raw * vis_mask).sum(dim=(1, 2, 3))
    pred_invisible_count = (pred_raw * mask).sum(dim=(1, 2, 3))
    gt_visible_count = (gt_for_count * vis_mask).sum(dim=(1, 2, 3))
    gt_invisible_count = (gt_for_count * mask).sum(dim=(1, 2, 3))

    if loss_mode == "density_mse_count_mse":
        vis_count_err = (pred_visible_count - gt_visible_count) ** 2
        inv_count_err = (pred_invisible_count - gt_invisible_count) ** 2
    else:
        vis_count_err = torch.abs(pred_visible_count - gt_visible_count)
        inv_count_err = torch.abs(pred_invisible_count - gt_invisible_count)

    vis_count_err = vis_count_err / gt_visible_count.clamp_min(1.0)
    inv_count_err = inv_count_err / gt_invisible_count.clamp_min(1.0)

    count_loss = (
        vis_count_err + float(invisible_count_weight) * inv_count_err
    ).mean()

    return density_loss + count_loss_weight * count_loss, density_loss

# -----------------------------
# Count helpers
# -----------------------------
def _counts_from_densities(
    pred_density,
    gt_density,
    density_scale: float,
    gt_downsample: str = "bilinear",
):
    """
    Derive predicted and GT counts, handling the resolution mismatch
    between CSRNet-style 1/8-res predictions and full-res GT.
    """
    gt_down, _ = _gt_density_resize(
        gt_density=gt_density,
        pred_density=pred_density,
        gt_downsample=gt_downsample,
        density_scale=density_scale,
    )
    gt_count = gt_down.sum(dim=(1, 2, 3))
    pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
    return pred_count, gt_count


def _split_counts_by_mask(
    pred_density: torch.Tensor,
    gt_density: torch.Tensor,
    mask: torch.Tensor | None,
    density_scale: float,
    gt_downsample: str = "bilinear",
):
    """
    Return (pred_visible, pred_invisible, gt_visible, gt_invisible) counts
    as (N,) tensors. ``mask`` is resized to pred resolution with nearest
    neighbor (matches loss path). When ``mask`` is None, invisible == 0.
    """
    gt_down, _ = _gt_density_resize(
        gt_density=gt_density,
        pred_density=pred_density,
        gt_downsample=gt_downsample,
        density_scale=density_scale,
    )
    if mask is None:
        m = torch.zeros_like(pred_density)
    else:
        if mask.shape[-2:] != pred_density.shape[-2:]:
            m = F.interpolate(mask, size=pred_density.shape[-2:], mode="nearest")
        else:
            m = mask
        m = m.clamp(0.0, 1.0)
    vis = 1.0 - m
    pred_raw = pred_density / density_scale
    pred_vis = (pred_raw * vis).sum(dim=(1, 2, 3))
    pred_inv = (pred_raw * m).sum(dim=(1, 2, 3))
    gt_vis = (gt_down * vis).sum(dim=(1, 2, 3))
    gt_inv = (gt_down * m).sum(dim=(1, 2, 3))
    return pred_vis, pred_inv, gt_vis, gt_inv


# -----------------------------
# Encoder-only inpainting auxiliary loss (per-patch density reconstruction)
# -----------------------------
def _compute_aux_hidden_density_loss(
    tokens: torch.Tensor,            # (B, N, D)  encoder patch tokens
    patch_grid: tuple[int, int],     # (h, w)     N == h*w
    aux_head: torch.nn.Module,       # nn.Linear(D, p*p)
    gt_density: torch.Tensor,        # (B, 1, H, W) raw, sum-preserving
    mask: torch.Tensor,              # (B, 1, H, W) in [0, 1]
    patch_size: int,
    patch_mask_threshold: float = 0.5,
    density_scale: float = DEFAULT_DENSITY_SCALE,
    count_loss_weight: float = 1.0,
) -> torch.Tensor:
    """
    Encoder-only inpainting auxiliary loss (per-patch density reconstruction).

    For each patch whose mask coverage exceeds ``patch_mask_threshold``, the
    aux head consumes the SINGLE encoder token at that position and outputs a
    flat p*p vector that we reshape into a p x p sub-patch density map. The
    loss is a weighted MSE against GT density, weighted by the per-pixel
    ``mask`` value so only the *hidden* sub-region of each masked patch is
    supervised (continuous in [0,1] for Gaussian masks, binary for box masks).
    Because the predictor is a single linear layer with no spatial mixing
    across tokens and no decoder in the path, gradient can only reduce by
    pushing hidden-density information into the encoder token at the masked
    position -- closes the "decoder smoothes the hole" shortcut.

    Returns an auxiliary scalar:
        hidden_density_mse + count_loss_weight * hidden_count_l1
    where density supervision uses the same scaled-target convention as the
    main density loss and hidden count is measured on the same supervised
    region, in raw "heads" units with relative normalization.
    Safely returns 0 (still on the autograd graph through ``aux_head``) when
    no patch in the batch is sufficiently masked.
    """
    B, N, D = tokens.shape
    h, w = patch_grid
    if N != h * w:
        raise ValueError(
            f"aux loss: tokens N={N} but patch grid {h}x{w} = {h*w}"
        )
    p = patch_size
    H_pad, W_pad = h * p, w * p
    H, W = gt_density.shape[-2:]

    # The encoder reflect-pads the input image to a multiple of ``p``; for the
    # AUX target we want zero density and zero mask in the padded region so
    # reflected density does not leak into the per-patch supervision.
    if (H, W) != (H_pad, W_pad):
        pad_h_amt = max(0, H_pad - H)
        pad_w_amt = max(0, W_pad - W)
        if pad_h_amt > 0 or pad_w_amt > 0:
            gt_density = F.pad(gt_density, (0, pad_w_amt, 0, pad_h_amt), value=0.0)
            mask = F.pad(mask, (0, pad_w_amt, 0, pad_h_amt), value=0.0)
        if H > H_pad or W > W_pad:
            gt_density = gt_density[..., :H_pad, :W_pad]
            mask = mask[..., :H_pad, :W_pad]

    mask = mask.clamp(0.0, 1.0)

    # Per-token p*p prediction -> spatial sub-patch density map.
    # Pred layout: (B, N, p*p) with N = h*w in row-major (token i = (hi, wi)).
    # Spatial layout we want: (B, 1, h*p, w*p) so that pixel (y, x) =
    # patch (y//p, x//p) at offset (y%p, x%p). Reshape via:
    #   (B, h, w, p, p) -> permute to (B, h, p, w, p) -> view (B, 1, h*p, w*p).
    pred = aux_head(tokens)                          # (B, N, p*p)
    pred = pred.view(B, h, w, p, p)
    pred = pred.permute(0, 1, 3, 2, 4).contiguous()  # (B, h, p, w, p)
    pred = pred.view(B, 1, h * p, w * p)

    # Patch-level gate: which patches count as "masked" for supervision.
    patch_mask_frac = F.avg_pool2d(mask, p)                       # (B, 1, h, w)
    patch_sel = (patch_mask_frac > patch_mask_threshold).to(mask.dtype)
    pixel_patch_sel = F.interpolate(patch_sel, scale_factor=p, mode="nearest")

    # Hidden-only supervision: weight = patch gate * per-pixel mask. This is
    # 0 outside masked patches, and inside masked patches it is the mask
    # value at that pixel (continuous for Gaussian masks, binary for box).
    weight = pixel_patch_sel * mask
    gt_target = gt_density * density_scale
    sse = ((pred - gt_target) ** 2 * weight).sum()
    # ``clamp(min=1.0)`` makes empty-batch behaviour numerically safe: if
    # ``weight.sum() == 0`` then ``sse`` is also 0, so the loss is 0 with
    # gradient flowing through ``pred`` (and hence through aux_head).
    denom = weight.sum().clamp(min=1.0)
    density_aux = sse / denom

    pred_raw = pred / density_scale
    pred_hidden_count = (pred_raw * weight).sum(dim=(1, 2, 3))
    gt_hidden_count = (gt_density * weight).sum(dim=(1, 2, 3))
    count_aux = (
        torch.abs(pred_hidden_count - gt_hidden_count)
        / gt_hidden_count.clamp_min(1.0)
    ).mean()

    return density_aux + float(count_loss_weight) * count_aux


# -----------------------------
# Train for one epoch
# -----------------------------
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    count_loss_weight: float = 1.0,
    loss_mode: str = "density_mse_count_l1",
    invisible_density_weight: float = 1.0,
    invisible_count_weight: float = 1.0,
    invisible_density_norm: str = "region_mean",
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    mask_mode: str | None = None,
    mask_fill: str = "imagenet_mean",
    aux_loss_weight: float = 0.0,
    aux_patch_threshold: float = 0.5,
    max_grad_norm: float | None = 1.0,
):

    model.train()

    aux_head = getattr(model, "aux_head", None)
    use_aux = (
        aux_loss_weight > 0.0
        and aux_head is not None
        and getattr(model, "hidden_count_aux", False)
        and mask_mode == "inpaint"
    )
    aux_patch_size = int(getattr(model, "patch_size", 16)) if use_aux else 0

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_mae_vis = 0.0
    total_mae_inv = 0.0
    total_aux_loss = 0.0
    aux_samples = 0  # batches that contributed a non-zero aux loss
    inv_samples = 0  # only count samples where mask is actually non-empty
    total_samples = 0

    for batch in tqdm(dataloader, desc="train"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)
        mask = batch["mask"].to(device)

        images = apply_mask_fill(
            images, mask, mask_fill,
            mask_token=getattr(model, "mask_token", None),
        )

        optimizer.zero_grad()

        if use_aux:
            pred_density, tokens, patch_grid = model(images, return_tokens=True)
        else:
            pred_density = model(images)

        loss, mse = compute_loss(
            pred_density,
            gt_density,
            count_loss_weight=count_loss_weight,
            invisible_density_weight=invisible_density_weight,
            invisible_count_weight=invisible_count_weight,
            loss_mode=loss_mode,
            density_scale=density_scale,
            gt_downsample=gt_downsample,
            mask=mask,
            mask_mode=mask_mode,
            invisible_density_norm=invisible_density_norm,
        )

        aux_loss_value = 0.0
        if use_aux:
            aux_loss = _compute_aux_hidden_density_loss(
                tokens=tokens,
                patch_grid=patch_grid,
                aux_head=aux_head,
                gt_density=gt_density,
                mask=mask,
                patch_size=aux_patch_size,
                patch_mask_threshold=aux_patch_threshold,
                density_scale=density_scale,
                count_loss_weight=count_loss_weight,
            )
            loss = loss + aux_loss_weight * aux_loss
            aux_loss_value = float(aux_loss.detach().item())
        with torch.no_grad():
            pred_count, gt_count = _counts_from_densities(
                pred_density, gt_density, density_scale, gt_downsample=gt_downsample
            )
            pred_vis, pred_inv, gt_vis, gt_inv = _split_counts_by_mask(
                pred_density, gt_density, mask, density_scale, gt_downsample=gt_downsample
            )
            # Only include samples whose mask is non-empty in the invisible MAE
            # so "train_mae_invisible" is not artificially diluted by zero-mask
            # crops in density-biased batches.
            mask_has_any = mask.flatten(1).sum(dim=1) > 0
            inv_abs = torch.abs(pred_inv - gt_inv)
            vis_abs = torch.abs(pred_vis - gt_vis)
        mae = torch.abs(pred_count - gt_count).sum()

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_mae += mae.item()
        total_mae_vis += vis_abs.sum().item()
        if mask_has_any.any():
            total_mae_inv += inv_abs[mask_has_any].sum().item()
            inv_samples += int(mask_has_any.sum().item())
        if use_aux:
            total_aux_loss += aux_loss_value * batch_size
            aux_samples += batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_mae_vis = total_mae_vis / total_samples
    avg_mae_inv = total_mae_inv / inv_samples if inv_samples > 0 else 0.0
    avg_aux_loss = total_aux_loss / aux_samples if aux_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "mae": avg_mae,
        "mse": avg_mse,
        "mae_visible": avg_mae_vis,
        "mae_invisible": avg_mae_inv,
        "invisible_sample_count": inv_samples,
        "aux_loss": avg_aux_loss,
        "aux_active": bool(use_aux),
    }


# -----------------------------
# Tiled inference
# -----------------------------
def _pad_to_tileable(dim: int, tile_size: int, stride: int) -> int:
    """Smallest ``T >= max(dim, tile_size)`` such that ``(T - tile_size) % stride == 0``."""
    if dim <= tile_size:
        return tile_size
    excess = (dim - tile_size) % stride
    return dim + ((stride - excess) % stride)


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int = 224,
    overlap: int = 48,
    max_batch: int = 16,
) -> torch.Tensor:
    """
    Sliding-window prediction with Hann-window blending.

    Splits ``image`` (C, H, W) into overlapping ``tile_size x tile_size`` tiles with
    stride ``tile_size - overlap``, runs them through ``model`` in batches of up to
    ``max_batch``, and stitches the outputs using a 2D Hann window. The returned density
    has the same spatial size ``(H, W)`` as ``image``.

    Count preservation: each pixel's output is a window-weighted average of all tile
    predictions covering it, so ``out.sum()`` is the sum of per-tile predictions
    weighted by ``window / total_window`` -- same as summing non-overlapping predictions
    in expectation. Combined with non-zero ``clamp_min`` on the windows, edge pixels
    covered by only one tile get that tile's value unchanged.

    Use this at val time so the ViT sees tiles at the **exact** resolution it was
    trained on, instead of a huge ``dynamic_img_size`` interpolation of positional
    embeddings to the full image resolution.
    """
    if image.ndim != 3:
        raise ValueError(f"predict_tiled expects (C,H,W), got {tuple(image.shape)}")
    C, H, W = image.shape
    device = image.device
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"Require tile_size > overlap >= 0, got tile_size={tile_size}, overlap={overlap}"
        )
    stride = tile_size - overlap

    Hp = _pad_to_tileable(H, tile_size, stride)
    Wp = _pad_to_tileable(W, tile_size, stride)
    pad_bottom = Hp - H
    pad_right = Wp - W
    if pad_bottom > 0 or pad_right > 0:
        # Reflect-pad requires pad < dim along each axis; fall back to zero-pad
        # for pathologically small images (much smaller than a tile).
        mode = "reflect" if (pad_bottom < H and pad_right < W) else "constant"
        img = F.pad(image.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode=mode).squeeze(0)
    else:
        img = image

    # Enumerate tile top-left positions.
    coords: list[tuple[int, int]] = []
    for top in range(0, Hp - tile_size + 1, stride):
        for left in range(0, Wp - tile_size + 1, stride):
            coords.append((top, left))

    # 2D Hann window. ``clamp_min`` keeps pixels covered by only one tile well-defined.
    wh = torch.hann_window(tile_size, periodic=False, device=device, dtype=torch.float32)
    ww = torch.hann_window(tile_size, periodic=False, device=device, dtype=torch.float32)
    window = (wh[:, None] * ww[None, :]).clamp_min(1e-3)

    # Accumulators in float32 to avoid bf16/fp16 rounding on long sums.
    out_sum = torch.zeros(1, Hp, Wp, device=device, dtype=torch.float32)
    w_sum = torch.zeros(1, Hp, Wp, device=device, dtype=torch.float32)

    for start in range(0, len(coords), max_batch):
        chunk = coords[start : start + max_batch]
        batch = torch.stack(
            [img[:, t : t + tile_size, l : l + tile_size] for (t, l) in chunk],
            dim=0,
        )
        pred = model(batch)  # (N, 1, tile, tile)
        pred = pred[:, 0].float()  # (N, tile, tile)
        for k, (t, l) in enumerate(chunk):
            out_sum[0, t : t + tile_size, l : l + tile_size] += pred[k] * window
            w_sum[0, t : t + tile_size, l : l + tile_size] += window

    out = (out_sum / w_sum)[:, :H, :W]
    return out.unsqueeze(0)  # (1, 1, H, W)


@torch.no_grad()
def extract_vit_latent_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int = 224,
    overlap: int = 48,
    max_batch: int = 16,
) -> torch.Tensor:
    """
    Extract one ViT latent vector from a full-resolution image using tiled inference.

    Uses the same tiling setup as :func:`predict_tiled` but, instead of decoding density,
    it pools encoder tokens per tile and then averages tile latents.
    """
    if image.ndim != 3:
        raise ValueError(f"extract_vit_latent_tiled expects (C,H,W), got {tuple(image.shape)}")
    if not hasattr(model, "encoder") or not hasattr(model, "num_prefix_tokens"):
        raise ValueError("extract_vit_latent_tiled requires a ViT-style model with encoder.")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"Require tile_size > overlap >= 0, got tile_size={tile_size}, overlap={overlap}"
        )

    C, H, W = image.shape
    device = image.device
    stride = tile_size - overlap

    Hp = _pad_to_tileable(H, tile_size, stride)
    Wp = _pad_to_tileable(W, tile_size, stride)
    pad_bottom = Hp - H
    pad_right = Wp - W
    if pad_bottom > 0 or pad_right > 0:
        mode = "reflect" if (pad_bottom < H and pad_right < W) else "constant"
        img = F.pad(image.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode=mode).squeeze(0)
    else:
        img = image

    coords: list[tuple[int, int]] = []
    for top in range(0, Hp - tile_size + 1, stride):
        for left in range(0, Wp - tile_size + 1, stride):
            coords.append((top, left))

    tile_latents = []
    for start in range(0, len(coords), max_batch):
        chunk = coords[start : start + max_batch]
        batch = torch.stack(
            [img[:, t : t + tile_size, l : l + tile_size] for (t, l) in chunk],
            dim=0,
        )
        tokens = model.encoder.forward_features(batch)
        tokens = tokens[:, model.num_prefix_tokens :, :]
        tile_latents.append(tokens.mean(dim=1))

    return torch.cat(tile_latents, dim=0).mean(dim=0)


# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    tiled: bool = False,
    tile_size: int = 224,
    tile_overlap: int = 48,
    tile_max_batch: int = 16,
):
    """
    Per-image MAE on counts.

    If ``tiled=True``, predictions are produced by :func:`predict_tiled` at the image's
    native resolution. This matches the training-time tile size exactly and removes
    the attention-length shift introduced by feeding full images to a model that was
    trained on smaller crops.

    Tiled mode requires dataloader batches of size 1 (every val image has a different
    native shape). The non-tiled path still supports batched eval with same-shape images.
    """

    model.eval()

    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="val"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        if tiled:
            if images.size(0) != 1:
                raise RuntimeError(
                    f"tiled validation requires val batch size 1, got {images.size(0)}."
                )
            pred_density = predict_tiled(
                model,
                images[0],
                tile_size=tile_size,
                overlap=tile_overlap,
                max_batch=tile_max_batch,
            )
        else:
            pred_density = model(images)

        pred_count, gt_count = _counts_from_densities(
            pred_density, gt_density, density_scale, gt_downsample=gt_downsample
        )
        mae = torch.abs(pred_count - gt_count).sum()

        total_mae += mae.item()
        total_samples += images.size(0)

    return total_mae / total_samples


@torch.no_grad()
def validate_masked(
    model,
    dataloader,
    device,
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    tiled: bool = False,
    tile_size: int = 224,
    tile_overlap: int = 48,
    tile_max_batch: int = 16,
    mask_fill: str = "imagenet_mean",
) -> dict:
    """
    Hallucination-aware validation on a dataset built with ``mask_mode="inpaint"``
    and ``mask_object_ratio > 0``:

    - ``mae_total_masked``: |sum(pred) - sum(gt_full)|  -- total count when the
      model is shown a masked input; gt stays full because inpaint mode does not
      zero density inside the mask.
    - ``mae_hidden``: |sum(pred * mask) - sum(gt * mask)| -- how well the model
      hallucinates density inside the holes. This is the direct test of the
      hypothesis.
    - ``mae_visible_masked``: |sum(pred * (1-mask)) - sum(gt * (1-mask))| --
      sanity check that the visible region is not regressed by the inpainting task.
    - ``mean_mask_fraction``: average of ``mask.mean()`` per image, useful to sanity
      check that the masked val loader actually has non-empty masks.
    """
    model.eval()

    total_mae_total = 0.0
    total_mae_hidden = 0.0
    total_mae_visible = 0.0
    total_mask_frac = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="val-masked"):
        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)
        mask = batch["mask"].to(device)

        if mask_fill != "imagenet_mean":
            images = apply_mask_fill(
                images, mask, mask_fill,
                mask_token=getattr(model, "mask_token", None),
            )

        if tiled:
            if images.size(0) != 1:
                raise RuntimeError(
                    f"tiled validation requires val batch size 1, got {images.size(0)}."
                )
            pred_density = predict_tiled(
                model,
                images[0],
                tile_size=tile_size,
                overlap=tile_overlap,
                max_batch=tile_max_batch,
            )
        else:
            pred_density = model(images)

        pred_count, gt_count = _counts_from_densities(
            pred_density, gt_density, density_scale, gt_downsample=gt_downsample
        )
        pred_vis, pred_inv, gt_vis, gt_inv = _split_counts_by_mask(
            pred_density, gt_density, mask, density_scale, gt_downsample=gt_downsample
        )
        total_mae_total += torch.abs(pred_count - gt_count).sum().item()
        total_mae_hidden += torch.abs(pred_inv - gt_inv).sum().item()
        total_mae_visible += torch.abs(pred_vis - gt_vis).sum().item()
        total_mask_frac += float(mask.flatten(1).mean(dim=1).sum().item())
        total_samples += images.size(0)

    n = max(total_samples, 1)
    return {
        "mae_total_masked": total_mae_total / n,
        "mae_hidden": total_mae_hidden / n,
        "mae_visible_masked": total_mae_visible / n,
        "mean_mask_fraction": total_mask_frac / n,
    }


# ---------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------
def plot_training_curves(history, save_path: str | Path):
    """
    Two-panel plot:
    - Top: training losses / MAE (visible / invisible / total).
    - Bottom: validation MAE (clean total, masked total, hidden, visible-masked).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    has_masked = any(len(history.get(k, [])) > 0 for k in (
        "val_mae_total_masked", "val_mae_hidden", "val_mae_visible_masked"
    ))

    fig, axes = plt.subplots(2 if has_masked else 1, 1, figsize=(9, 9 if has_masked else 5), sharex=True)
    if not has_masked:
        axes = [axes]

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["train_mae"], label="Train MAE (total)")
    if history.get("train_mae_visible"):
        ax.plot(epochs, history["train_mae_visible"], label="Train MAE (visible)", linestyle="--")
    if history.get("train_mae_invisible"):
        ax.plot(epochs, history["train_mae_invisible"], label="Train MAE (invisible)", linestyle="--")
    if history["val_mae"]:
        ax.plot(epochs, history["val_mae"], label="Val MAE (clean)", color="black")
    aux_hist = history.get("train_aux_loss") or []
    if aux_hist and any(v != 0.0 for v in aux_hist):
        # Aux loss lives on a different scale (weighted MSE on hidden-region
        # density per masked patch), so plot on a twin y-axis to keep the
        # main MAE / total-loss curves readable.
        ax_aux = ax.twinx()
        ax_aux.plot(epochs, aux_hist, color="tab:purple", linestyle="-.", label="Train aux loss (hidden-density)")
        ax_aux.set_ylabel("Aux loss")
        ax_aux.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("Metric")
    ax.set_title("Training Curves")
    ax.legend(loc="best", fontsize=8)

    if has_masked:
        ax2 = axes[1]
        if history.get("val_mae_total_masked"):
            ax2.plot(epochs, history["val_mae_total_masked"], label="Val MAE total (masked input)")
        if history.get("val_mae_hidden"):
            ax2.plot(epochs, history["val_mae_hidden"], label="Val MAE hidden (hallucination)")
        if history.get("val_mae_visible_masked"):
            ax2.plot(epochs, history["val_mae_visible_masked"], label="Val MAE visible (masked input)")
        if history.get("val_mae"):
            ax2.plot(epochs, history["val_mae"], label="Val MAE (clean)", color="black", linestyle=":")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.set_title("Hallucination validation (masked input)")
        ax2.legend(loc="best", fontsize=8)
    else:
        axes[0].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()


# ---------------------------------------------------------

# Main training function
# -----------------------------
def train(
    model,
    train_dataset,
    val_dataset=None,
    masked_val_dataset=None,
    epochs: int = 400,
    batch_size: int = 8,
    count_loss_weight: float = 1.0,
    invisible_density_weight: float = 1.0,
    invisible_count_weight: float = 1.0,
    invisible_density_norm: str = "region_mean",
    loss_mode: str = "density_mse_count_l1",
    early_stopping_patience: int | None = None,
    model_name: str = "model",
    data_name: str = "data",
    mask_ratio: float | None = None,
    mask_mode: str | None = None,
    mask_dot_style: str | None = None,
    mask_sampling_mode: str = "random",
    mask_fill: str = "imagenet_mean",
    output_dir: str | Path = "trained_models",
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    validate_during_training: bool = True,
    optimizer_type: str = "sgd",
    lr: float = 1e-7,
    momentum: float = 0.95,
    weight_decay: float = 5e-4,
    unfreeze_backbone_after_epoch: int | None = None,
    max_grad_norm: float | None = 1.0,
    resume_checkpoint: str | Path | None = None,
    log_dir: str | Path | None = None,
    val_batch_size: int | None = None,
    tiled_val: bool = False,
    tiled_val_tile: int = 224,
    tiled_val_overlap: int = 48,
    tiled_val_max_batch: int = 16,
    aux_loss_weight: float = 0.0,
    aux_patch_threshold: float = 0.5,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Training {model_name} model on {data_name} dataset on",
        device,
        f"with {epochs} epochs, batch size: {batch_size}, loss mode: {loss_mode}, "
        f"count loss weight: {count_loss_weight}, density scale: {density_scale}, "
        f"invisible density weight: {invisible_density_weight}, "
        f"invisible count weight: {invisible_count_weight}, "
        f"gt_downsample: {gt_downsample}, "
        f"mask ratio: {mask_ratio}, mask mode: {mask_mode}, mask dot style: {mask_dot_style or 'box'}, "
        f"mask fill: {mask_fill}, "
        f"optimizer: {optimizer_type}, lr: {lr}"
    )
    if mask_fill not in MASK_FILL_MODES:
        raise ValueError(
            f"mask_fill must be one of {MASK_FILL_MODES}, got {mask_fill!r}."
        )
    if mask_fill == "learnable" and not hasattr(model, "mask_token"):
        raise ValueError(
            "mask_fill='learnable' requires the model to expose a `mask_token` "
            "parameter of shape (3,). ViTDensity provides it by default."
        )

    if unfreeze_backbone_after_epoch is not None:
        if not hasattr(model, "encoder"):
            print("Warning: unfreeze_backbone_after_epoch is set but model has no "
            "`encoder`; scheduled unfreeze will be skipped."
            )
        else:
            print( f"Backbone (`encoder`) will be unfrozen starting at epoch "
                    f"{unfreeze_backbone_after_epoch + 1} "
                    f"(after completing epoch {unfreeze_backbone_after_epoch})"
                    f"."
            )

    model = model.to(device)

    if False and isinstance(train_dataset, PatchAugmentedDataset):
        effective_batch_size = train_dataset.variants_per_image
        print(
            "Detected CSRNetPatchAugmentedDataset for training. "
            f"Using one image per batch ({effective_batch_size} patches)."
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PatchBatchSampler(train_dataset, shuffle=True),
            num_workers=8,
            pin_memory=True,
        )
        default_val_batch_size = 1
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
        )
        default_val_batch_size = batch_size

    # Caller can force a smaller val batch (e.g. native-resolution ViT eval where every
    # image has a different shape, so batching is not possible).
    if val_batch_size is None:
        val_batch_size = default_val_batch_size

    val_loader = None
    masked_val_loader = None
    if validate_during_training:
        if val_dataset is None:
            raise ValueError(
                "val_dataset must be provided when validate_during_training=True."
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        if masked_val_dataset is not None:
            masked_val_loader = DataLoader(
                masked_val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
    elif early_stopping_patience is not None:
        print("Validation disabled: early stopping requires validation MAE and will be ignored.")

    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable parameters (check freeze_encoder / masking).")

    start_epoch = 1
    completed_epoch = 0
    ckpt: dict | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        # Tolerate missing keys that were added after this checkpoint was saved
        # (e.g. ``mask_token``, ``aux_head.*``). Unexpected keys are tolerated
        # only when they belong to optional, opt-in heads that the current
        # model does not include (so resuming a hidden-count-aux checkpoint
        # into a non-aux model still works for representation probing).
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        unexpected_optional = {"mask_token"}
        unexpected_real = [
            k for k in unexpected
            if k not in unexpected_optional and not k.startswith("aux_head.")
        ]
        if unexpected_real:
            raise RuntimeError(
                f"Unexpected keys in resume checkpoint {resume_path}: {unexpected_real[:8]}"
            )
        if unexpected:
            print(
                f"Resume {resume_path}: dropping checkpoint keys absent in current "
                f"model (e.g. aux head from a different config): {unexpected}."
            )
        if missing:
            print(
                f"Resume {resume_path}: missing keys initialized from scratch: {missing}."
            )
        completed_epoch = int(ckpt["epoch"])
        if (
            unfreeze_backbone_after_epoch is not None
            and hasattr(model, "encoder")
            and completed_epoch > unfreeze_backbone_after_epoch
        ):
            for p in model.encoder.parameters():
                p.requires_grad = True
        start_epoch = completed_epoch + 1
        print(
            f"Loaded checkpoint {resume_path} "
            f"(completed epoch {completed_epoch}); continuing from epoch {start_epoch}."
        )
        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint epoch {completed_epoch} already finished training (--epochs {epochs}); "
                "increase --epochs or use a different checkpoint."
            )

    backbone_lr = lr * 0.1
    param_groups = build_optimizer_param_groups(model, lr, backbone_lr, weight_decay)

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # warmup_epochs = 5
    warmup_epochs=unfreeze_backbone_after_epoch if unfreeze_backbone_after_epoch is not None else 5

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)

    lr_scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )

    best_mae = float("inf")
    epochs_without_improvement = 0
    history: dict = {
        "train_loss": [],
        "train_mae": [],
        "train_mse": [],
        "train_mae_visible": [],
        "train_mae_invisible": [],
        "train_aux_loss": [],
        "val_mae": [],
        "val_mae_total_masked": [],
        "val_mae_hidden": [],
        "val_mae_visible_masked": [],
        "val_mean_mask_fraction": [],
    }
    backbone_unfreeze_done = False
    count_weight_increase_done_1 = False
    count_weight_increase_done_2 = False

    if ckpt is not None:
        if "lr_scheduler_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        else:
            print(
                "Warning: checkpoint has no lr_scheduler_state_dict (old format). "
                "Replaying scheduler steps to align state, then restoring optimizer tensors."
            )
            for _ in range(completed_epoch):
                lr_scheduler.step()
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "best_mae" in ckpt and ckpt["best_mae"] is not None:
            best_mae = ckpt["best_mae"]
        if "epochs_without_improvement" in ckpt:
            epochs_without_improvement = int(ckpt["epochs_without_improvement"])
        if "history" in ckpt and isinstance(ckpt["history"], dict):
            ck_hist = ckpt["history"]
            for k in history:
                history[k] = list(ck_hist.get(k, []))
        if "count_loss_weight" in ckpt:
            count_loss_weight = float(ckpt["count_loss_weight"])
        if ckpt.get("backbone_unfreeze_done") is True:
            backbone_unfreeze_done = True
        elif (
            unfreeze_backbone_after_epoch is not None
            and completed_epoch > unfreeze_backbone_after_epoch
        ):
            backbone_unfreeze_done = True
        if ckpt.get("count_weight_increase_done_1") is True:
            count_weight_increase_done_1 = True
        if ckpt.get("count_weight_increase_done_2") is True:
            count_weight_increase_done_2 = True

    if mask_ratio is None:
        mask_str = "nomsk"
    else:
        mask_str = f"{mask_ratio}-{mask_mode or 'inpaint'}"
        if mask_dot_style and mask_dot_style != "box":
            mask_str = f"{mask_str}-{mask_dot_style}"
        mask_str = f"{mask_str}-{mask_sampling_mode}"
    run_name = f"{model_name}-{data_name}-{mask_str}"
    if (
        aux_loss_weight > 0.0
        and getattr(model, "hidden_count_aux", False)
        and mask_mode == "inpaint"
    ):
        # Distinguish runs with the encoder-only auxiliary head so checkpoints,
        # curve plots, and history files don't clash with non-aux baselines.
        run_name = f"{run_name}-aux{aux_loss_weight:g}"
    
    # output_dir = Path(output_dir) / model_name / run_name
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = log_dir

    # Resolve base dataset + first sample index for val visualization.
    if val_loader is not None:
        _vds = val_loader.dataset
        if hasattr(_vds, "dataset") and hasattr(_vds, "indices"):
            _vis_base_ds = _vds.dataset
            _vis_first_idx = _vds.indices[0]
        else:
            _vis_base_ds = _vds
            _vis_first_idx = 0
    _mvis_base_ds = None
    _mvis_first_idx = 0
    if masked_val_loader is not None:
        _mds = masked_val_loader.dataset
        if hasattr(_mds, "dataset") and hasattr(_mds, "indices"):
            _mvis_base_ds = _mds.dataset
            _mvis_first_idx = _mds.indices[0]
        else:
            _mvis_base_ds = _mds
            _mvis_first_idx = 0

    count_weight_controller = CountWeightController(w_max=0.01, p=2.0, warmup_epochs=warmup_epochs)

    for epoch in range(start_epoch, epochs + 1):
        if(not backbone_unfreeze_done
            and unfreeze_backbone_after_epoch is not None
            and hasattr(model, "encoder")
            and epoch > unfreeze_backbone_after_epoch
        ):
            enc = model.encoder
            n_frozen = sum(1 for p in enc.parameters() if not p.requires_grad)
            for p in enc.parameters():
                p.requires_grad = True

            if n_frozen > 0:
                print(
                    f"Epoch {epoch:03d} | Unfrozen backbone (`encoder`); "
                    f"(backbone lr={backbone_lr:.2e})."
                )
            else:
                print(
                f"Epoch {epoch:03d} | Backbone unfreeze skipped: "
                f"encoder was already trainable."
            )
            backbone_unfreeze_done = True


        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            count_loss_weight=count_loss_weight,
            invisible_density_weight=invisible_density_weight,
            invisible_count_weight=invisible_count_weight,
            invisible_density_norm=invisible_density_norm,
            loss_mode=loss_mode,
            density_scale=density_scale,
            gt_downsample=gt_downsample,
            mask_mode=mask_mode,
            mask_fill=mask_fill,
            aux_loss_weight=aux_loss_weight,
            aux_patch_threshold=aux_patch_threshold,
            max_grad_norm=max_grad_norm,
        )
        train_loss = train_metrics["loss"]
        train_mae = train_metrics["mae"]
        train_mse = train_metrics["mse"]
        train_mae_vis = train_metrics["mae_visible"]
        train_mae_inv = train_metrics["mae_invisible"]
        inv_samp = train_metrics["invisible_sample_count"]
        train_aux_loss = train_metrics.get("aux_loss", 0.0)
        aux_active = train_metrics.get("aux_active", False)
        count_loss_weight = count_weight_controller.update(epoch, train_mse)         # Step the scheduler to update the LR
        print(f"Epoch {epoch:03d} | Count loss weight: {count_loss_weight:.4f}")
        lr_scheduler.step()
        
        val_mae = None
        masked_val_metrics = None
        if validate_during_training:
            val_mae = validate(
                model,
                val_loader,
                device,
                density_scale=density_scale,
                gt_downsample=gt_downsample,
                tiled=tiled_val,
                tile_size=tiled_val_tile,
                tile_overlap=tiled_val_overlap,
                tile_max_batch=tiled_val_max_batch,
            )
            if masked_val_loader is not None:
                masked_val_metrics = validate_masked(
                    model,
                    masked_val_loader,
                    device,
                    density_scale=density_scale,
                    gt_downsample=gt_downsample,
                    tiled=tiled_val,
                    tile_size=tiled_val_tile,
                    tile_overlap=tiled_val_overlap,
                    tile_max_batch=tiled_val_max_batch,
                    mask_fill=mask_fill,
                )

        uniq_lrs = sorted({g["lr"] for g in optimizer.param_groups}, reverse=True)
        current_lr = uniq_lrs[0] if len(uniq_lrs) == 1 else uniq_lrs
        lr_str = (
            f"{current_lr:.2e}"
            if not isinstance(current_lr, list)
            else "/".join(f"{x:.2e}" for x in current_lr)
        )



        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["train_mse"].append(train_mse)
        history["train_mae_visible"].append(train_mae_vis)
        history["train_mae_invisible"].append(train_mae_inv)
        history["train_aux_loss"].append(train_aux_loss)
        if val_mae is not None:
            history["val_mae"].append(val_mae)
        if masked_val_metrics is not None:
            history["val_mae_total_masked"].append(masked_val_metrics["mae_total_masked"])
            history["val_mae_hidden"].append(masked_val_metrics["mae_hidden"])
            history["val_mae_visible_masked"].append(masked_val_metrics["mae_visible_masked"])
            history["val_mean_mask_fraction"].append(masked_val_metrics["mean_mask_fraction"])

        plot_training_curves(history, output_dir / f"{run_name}-curves.png")

        # Save latest checkpoint every epoch (for resuming).
        latest_path = output_dir / f"{run_name}-latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "best_mae": best_mae,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "count_loss_weight": count_loss_weight,
            "backbone_unfreeze_done": backbone_unfreeze_done,
            "count_weight_increase_done_1": count_weight_increase_done_1,
            "count_weight_increase_done_2": count_weight_increase_done_2,
        }, latest_path)

        if val_mae is not None:
            split_str = (
                f"Train MAE vis/inv {train_mae_vis:.4f}/{train_mae_inv:.4f} "
                f"(inv over {inv_samp} samples) | "
            )
            if masked_val_metrics is not None:
                m = masked_val_metrics
                masked_str = (
                    f"Val masked: total {m['mae_total_masked']:.4f} | "
                    f"hidden {m['mae_hidden']:.4f} | "
                    f"visible {m['mae_visible_masked']:.4f} | "
                    f"mask_frac {m['mean_mask_fraction']:.3f} | "
                )
            else:
                masked_str = ""
            aux_str = (
                f"aux_loss {train_aux_loss:.4f} (w={aux_loss_weight:g}) | "
                if aux_active else ""
            )
            print(
                f"Epoch {epoch:03d} | "
                f"lr {lr_str} | "
                f"Train MAE {train_mae:.4f} | "
                f"{split_str}"
                f"weighted train mae {train_mae*count_loss_weight:.4f} | "
                f"train mse {train_mse:.4f} | "
                f"total loss {train_loss:.4f} | "
                f"{aux_str}"
                f"Val MAE {val_mae:.4f} | "
                f"{masked_str}"
                f"Best MAE {best_mae:.4f}"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                model_path = output_dir / f"{run_name}-best.pth"
                torch.save(model.state_dict(), model_path)
                print(f" * New best MAE {best_mae:.2f} — saved to {model_path}")
                epochs_without_improvement = 0

                visualize_image_and_density(
                    _vis_base_ds,
                    index=_vis_first_idx,
                    use_precomputed_density=True,
                    pred_density_scale=density_scale,
                    save_dir=output_dir,
                    model=model,
                )
                # C3: also render one masked val sample at best-MAE so we can
                # visually track hallucination quality epoch-by-epoch.
                if _mvis_base_ds is not None:
                    try:
                        visualize_image_and_density(
                            _mvis_base_ds,
                            index=_mvis_first_idx,
                            use_precomputed_density=True,
                            pred_density_scale=density_scale,
                            save_dir=output_dir / "masked_vis",
                            model=model,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"Warning: masked visualization failed: {e}")
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement in val MAE for {early_stopping_patience} epochs)."
                )
                break
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"lr {lr_str} | "
                f"total loss {train_loss:.4f} | "
                f"Train MAE {train_mae:.2f}"
            )

    history_path = output_dir / f"{run_name}-history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    curves_path = output_dir / f"{run_name}-curves.png"
    plot_training_curves(history, curves_path)

    print("Training finished")
    print(f"History saved to {history_path}")
    print(f"Curves saved to {curves_path}")
