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


# -----------------------------
# Loss: density MSE + choice of count loss (MAE or MSE on count)
# -----------------------------
def compute_loss(
    pred_density,
    gt_density,
    count_loss_weight: float = 1.0,
    invisible_density_weight: float = 1.0,
    invisible_count_weight: float = 1.0,
    loss_mode: str = "density_mse_count_l1",
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    mask: torch.Tensor = None,
    mask_mode: str = "robust",
    eps: float = 1e-6,
):
    """
    Density + count loss with explicit visible / invisible supervision.

    Key features:
    - Area-correct density resizing
    - Mask-aware density MSE (normalized)
    - Region-normalized count loss (prevents bias from mask size)
    - Separate weighting for invisible regions
    """

    if gt_downsample not in ("bilinear", "csrnet_cubic"):
        raise ValueError(f"gt_downsample must be 'bilinear' or 'csrnet_cubic', got {gt_downsample!r}")

    H_pred, W_pred = pred_density.shape[-2:]

    # -----------------------------
    # Resize GT density (preserve count)
    # -----------------------------
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
            scale_h = H_gt / H_pred
            scale_w = W_gt / W_pred
            gt_resized = gt_resized * (scale_h * scale_w)
            gt_for_count = gt_resized
            gt_target = gt_resized * density_scale
    else:
        gt_for_count = gt_density
        gt_target = gt_density * density_scale

    # -----------------------------
    # visible objects only case (standard loss)
    # -----------------------------
    if mask is None or not mask.any() or mask_mode == "robust":
        density_loss = F.mse_loss(pred_density, gt_target)

        pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
        gt_count = gt_for_count.sum(dim=(1, 2, 3))

        if loss_mode == "density_mse_count_mse":
            count_loss = F.mse_loss(pred_count, gt_count)
        else:
            count_loss = F.l1_loss(pred_count, gt_count)

        return density_loss + count_loss_weight * count_loss

    # -----------------------------
    # inpaint mask handling
    # -----------------------------
    if mask.shape[-2:] != pred_density.shape[-2:]:
        mask = F.interpolate(mask, size=pred_density.shape[-2:], mode="nearest")

    mask = mask.clamp(0.0, 1.0)
    vis_mask = 1.0 - mask

    # -----------------------------
    # Density loss (region-normalized)
    # -----------------------------
    sq_err = (pred_density - gt_target) ** 2

    vis_err = (sq_err * vis_mask).sum(dim=(1, 2, 3))
    inv_err = (sq_err * mask).sum(dim=(1, 2, 3))

    vis_pix = vis_mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    inv_pix = mask.sum(dim=(1, 2, 3)).clamp_min(eps)

    inv_w_den = float(invisible_density_weight)

    density_loss = (
        (vis_err / vis_pix) +
        inv_w_den * (inv_err / inv_pix)
    ).mean()

    # -----------------------------
    # Count loss (region-normalized)
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

    # Normalize by region size → critical fix
    vis_count_err = vis_count_err / vis_pix
    inv_count_err = inv_count_err / inv_pix

    inv_w_cnt = float(invisible_count_weight)

    count_loss = (vis_count_err + inv_w_cnt * inv_count_err).mean()

    # -----------------------------
    # Final loss
    # -----------------------------
    return density_loss + count_loss_weight * count_loss

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
    if gt_downsample not in ("bilinear", "csrnet_cubic"):
        raise ValueError(f"gt_downsample must be 'bilinear' or 'csrnet_cubic', got {gt_downsample!r}")

    if pred_density.shape[-2:] != gt_density.shape[-2:]:
        H_pred, W_pred = pred_density.shape[-2:]
        if gt_downsample == "csrnet_cubic":
            gt_down = downsample_gt_csrnet_cubic(gt_density, H_pred, W_pred)
        else:
            H_gt, W_gt = gt_density.shape[-2:]
            gt_down = F.interpolate(
                gt_density,
                size=(H_pred, W_pred),
                mode="bilinear",
                align_corners=False,
            )
            spatial_scale = (H_gt / H_pred) * (W_gt / W_pred)
            gt_down = gt_down * spatial_scale
        gt_count = gt_down.sum(dim=(1, 2, 3))
    else:
        gt_count = gt_density.sum(dim=(1, 2, 3))

    pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
    return pred_count, gt_count


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
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    mask_mode: str | None = None,
    max_grad_norm: float | None = 1.0,
):

    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="train"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        pred_density = model(images)

        loss = compute_loss(
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
        )
        with torch.no_grad():
            pred_count, gt_count = _counts_from_densities(
                pred_density, gt_density, density_scale, gt_downsample=gt_downsample
            )
        mae = torch.abs(pred_count - gt_count).sum()

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size


        total_mae += mae.item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples

    return avg_loss, avg_mae


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
):

    model.eval()

    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="val"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        pred_density = model(images)

        pred_count, gt_count = _counts_from_densities(
            pred_density, gt_density, density_scale, gt_downsample=gt_downsample
        )
        mae = torch.abs(pred_count - gt_count).sum()

        total_mae += mae.item()
        total_samples += images.size(0)

    return total_mae / total_samples


# ---------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------
def plot_training_curves(history, save_path: str | Path):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    if history["val_mae"]:
        plt.plot(epochs, history["val_mae"], label="Validation MAE")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curves")
    plt.legend()

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
    epochs: int = 400,
    batch_size: int = 8,
    count_loss_weight: float = 1.0,
    invisible_density_weight: float = 1.0,
    invisible_count_weight: float = 1.0,
    loss_mode: str = "density_mse_count_l1",
    early_stopping_patience: int | None = None,
    model_name: str = "model",
    data_name: str = "data",
    mask_ratio: float | None = None,
    mask_mode: str | None = None,
    mask_dot_style: str | None = None,
    mask_sampling_mode: str = "random",
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
        f"optimizer: {optimizer_type}, lr: {lr}"
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
        val_batch_size=1
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
        )
        val_batch_size=batch_size

    val_loader = None
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
        model.load_state_dict(ckpt["model_state_dict"])
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
        "val_mae": [],
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
            history = {
                "train_loss": list(ckpt["history"].get("train_loss", [])),
                "train_mae": list(ckpt["history"].get("train_mae", [])),
                "val_mae": list(ckpt["history"].get("val_mae", [])),
            }
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


        if(epoch > 0 and  epoch%20 == 0 and count_loss_weight <= 0.01):
            count_loss_weight = count_loss_weight * 2
            print(f"Epoch {epoch:03d} | Count loss weight increased to {count_loss_weight:.4f}")
            count_weight_increase_done_1 = True
        # if(epoch > 100 and not count_weight_increase_done_2):
        #     count_loss_weight = count_loss_weight * 10
        #     print(f"Epoch {epoch:03d} | Count loss weight increased to {count_loss_weight:.4f}")
        #     count_weight_increase_done_2 = True

        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            count_loss_weight=count_loss_weight,
            invisible_density_weight=invisible_density_weight,
            invisible_count_weight=invisible_count_weight,
            loss_mode=loss_mode,
            density_scale=density_scale,
            gt_downsample=gt_downsample,
            mask_mode=mask_mode,
            max_grad_norm=max_grad_norm,
        )

        # Step the scheduler to update the LR
        lr_scheduler.step()
        
        val_mae = None
        if validate_during_training:
            val_mae = validate(
                model,
                val_loader,
                device,
                density_scale=density_scale,
                gt_downsample=gt_downsample,
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
        if val_mae is not None:
            history["val_mae"].append(val_mae)

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
            print(
                f"Epoch {epoch:03d} | "
                f"lr {lr_str} | "
                f"Train MAE {train_mae:.4f} | "
                f"weighted train mae {train_mae*count_loss_weight:.4f} | "
                f"train mse {(train_loss - train_mae*count_loss_weight):.4f} | "
                f"total loss {train_loss:.4f} | "
                f"Val MAE {val_mae:.4f} | "
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
