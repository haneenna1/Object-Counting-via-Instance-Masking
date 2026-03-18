"""
Minimal training script for density map regression (object counting)
with logging and training curves.

Dataset yields raw density (sum ≈ count). Training uses density_scale only here:
  - MSE target = raw_gt * density_scale (stronger gradients)
  - pred_count = pred.sum() / density_scale, gt_count = raw_gt.sum()
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Default training scale (only applied in this module, not in dataset).
DEFAULT_DENSITY_SCALE = 255.0


# -----------------------------
# Loss: density MSE + choice of count loss (MAE or MSE on count)
# -----------------------------
def compute_loss(
    pred_density,
    gt_density,
    count_loss_weight: float = 1.0,
    loss_mode: str = "density_mse_count_l1",
    density_scale: float = DEFAULT_DENSITY_SCALE,
):
    """
    gt_density: raw from dataloader (integral ≈ count).
    Network is trained to match gt_density * density_scale.
    """
    # CSRNet-style: downsample raw GT, preserve integral
    if pred_density.shape[-2:] != gt_density.shape[-2:]:
        H_gt, W_gt = gt_density.shape[-2:]
        H_pred, W_pred = pred_density.shape[-2:]
        gt_density = F.interpolate(
            gt_density,
            size=(H_pred, W_pred),
            mode="bilinear",
            align_corners=False,
        )
        scale_h = H_gt / H_pred
        scale_w = W_gt / W_pred
        gt_density = gt_density * (scale_h * scale_w)

    gt_target = gt_density * density_scale
    density_loss = F.mse_loss(pred_density, gt_target)

    pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
    gt_count = gt_density.sum(dim=(1, 2, 3))

    if loss_mode == "density_mse_count_mse":
        count_loss = F.mse_loss(pred_count, gt_count)
    else:
        count_loss = F.l1_loss(pred_count, gt_count)

    return density_loss + count_loss_weight * count_loss


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
    density_scale: float = DEFAULT_DENSITY_SCALE,
):

    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="train"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        optimizer.zero_grad()

        pred_density = model(images)

        loss = compute_loss(
            pred_density,
            gt_density,
            count_loss_weight=count_loss_weight,
            loss_mode=loss_mode,
            density_scale=density_scale,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
        gt_count = gt_density.sum(dim=(1, 2, 3))

        mae = torch.abs(pred_count - gt_count).sum()

        total_mae += mae.item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples

    return avg_loss, avg_mae


# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate(model, dataloader, device, density_scale: float = DEFAULT_DENSITY_SCALE):

    model.eval()

    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="val"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        pred_density = model(images)

        pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
        gt_count = gt_density.sum(dim=(1, 2, 3))

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
    val_dataset,
    epochs: int = 100,
    batch_size: int = 8,
    count_loss_weight: float = 1.0,
    loss_mode: str = "density_mse_count_l1",
    early_stopping_patience: int | None = 15,
    model_name: str = "model",
    data_name: str = "data",
    mask_ratio: float | None = None,
    output_dir: str | Path = "trained_models",
    density_scale: float = DEFAULT_DENSITY_SCALE,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Training {model_name} model on {data_name} dataset on",
        device,
        f"with {epochs} epochs: , batch size: {batch_size}, loss mode: {loss_mode}, count loss weight: {count_loss_weight}, density scale: {density_scale}"
    )

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_str = "nomsk" if mask_ratio is None else str(mask_ratio)
    run_name = f"{model_name}-{data_name}-{mask_str}"

    best_mae = float("inf")
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    for epoch in range(1, epochs + 1):

        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            count_loss_weight=count_loss_weight,
            loss_mode=loss_mode,
            density_scale=density_scale,
        )

        val_mae = validate(
            model,
            val_loader,
            device,
            density_scale=density_scale,
        )

        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        plot_training_curves(history, output_dir / f"{run_name}-curves.png")

        print(
            f"Epoch {epoch:03d} | "
            f"total loss {train_loss:.4f} | "
            f"Train MAE {train_mae:.2f} | "
            f"weighted train mae {train_mae * count_loss_weight:.4f} | "
            f"train density mse {train_loss - train_mae * count_loss_weight:.4f} | "
            f"Val MAE {val_mae:.2f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            model_path = output_dir / f"{run_name}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement in val MAE for {early_stopping_patience} epochs)."
            )
            break

    history_path = output_dir / f"{run_name}-history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    curves_path = output_dir / f"{run_name}-curves.png"
    plot_training_curves(history, curves_path)

    print("Training finished")
    print(f"History saved to {history_path}")
    print(f"Curves saved to {curves_path}")
