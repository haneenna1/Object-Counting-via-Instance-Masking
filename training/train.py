"""
Minimal training script for density map regression (object counting)
with logging and training curves.

Density maps are in [0, DENSITY_SCALE] (e.g. 0-255); counts = density.sum() / DENSITY_SCALE.
"""

import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DENSITY_SCALE


# -----------------------------
# Loss: density MSE + optional L1 count loss (reduces MAE on total count)
# -----------------------------
def compute_loss(pred_density, gt_density, count_loss_weight=1.0):
    """
    Combined loss for density map regression.
    - MSE on density (both in [0, DENSITY_SCALE]): keeps spatial structure.
    - L1 on counts: |pred_count - gt_count| with count = density.sum() / DENSITY_SCALE.
    """
    density_mse = F.mse_loss(pred_density, gt_density)

    pred_count = pred_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE
    gt_count = gt_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE
    count_l1 = F.l1_loss(pred_count, gt_count)

    loss = density_mse + count_loss_weight * count_l1
    return loss


# -----------------------------
# Train for one epoch
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, device, count_loss_weight=1.0):

    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="train"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        optimizer.zero_grad()

        pred_density = model(images)

        loss = compute_loss(pred_density, gt_density, count_loss_weight=count_loss_weight)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        # Counts from density (scale back to 0-1 range for count)
        pred_count = pred_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE
        gt_count = gt_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE

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
def validate(model, dataloader, device):

    model.eval()

    total_mae = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="val"):

        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        pred_density = model(images)

        pred_count = pred_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE
        gt_count = gt_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE

        mae = torch.abs(pred_count - gt_count).sum()

        total_mae += mae.item()
        total_samples += images.size(0)

    return total_mae / total_samples


# ---------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------
def plot_training_curves(history):

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
    plt.savefig("training_curves.png")
    plt.close()


# ---------------------------------------------------------
# Main training function
# -----------------------------
def train(model, train_dataset, val_dataset, epochs=100, batch_size=8, count_loss_weight=1.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "Training on",
        device,
        "| epochs:",
        epochs,
        "| batch size:",
        batch_size,
        "| count_loss_weight:",
        count_loss_weight,
    )

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_mae = float("inf")

    # training history
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    # -----------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------
    for epoch in range(1, epochs + 1):

        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            count_loss_weight=count_loss_weight,
        )

        val_mae = validate(
            model,
            val_loader,
            device
        )

        # store metrics
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {train_loss:.4f} | "
            f"Train MAE {train_mae:.2f} | "
            f"Val MAE {val_mae:.2f}"
        )

        # save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # -----------------------------------------------------
    # Plot curves
    # -----------------------------------------------------
    plot_training_curves(history)

    print("Training finished")
    print("History saved to training_history.json")
    print("Curves saved to training_curves.png")