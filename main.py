from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sympy import N
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import kagglehub


from data import density
from data.dataset import (
    DENSITY_MAP_DIR_AUTO,
    precompute_density_maps,
    visualize_image_and_density,
)
from data.shanghaitech import load_shanghaitech_dataset
from data.fsc147 import visualize_fsc147_density, load_fsc147_dataset
from data.transforms import (
    random_crop_transform,
    resize_transform,
    horizontal_flip_transform,
    random_90deg_rotation_transform,
    color_jitter_transform,
    compose_transforms,
)
from model.unet import UNetDensity
from model.csrnet import CSRNet
from training.train import train, DEFAULT_DENSITY_SCALE

def visualize_sample(sample, model=None, density_scale: float = DEFAULT_DENSITY_SCALE):

    image = sample["image"].cpu().numpy().transpose(1, 2, 0)
    original_image = sample["original_image"].cpu().numpy().transpose(1, 2, 0)
    density = sample["density"].cpu().numpy().squeeze()
    mask = sample["mask"].cpu().numpy().squeeze()
    gt_count = sample["count"].item()  # ≈ density.sum() (raw GT)

    fig, axs = plt.subplots(1, 5, figsize=(18, 5))

    # image
    axs[0].imshow(original_image)
    axs[0].set_title(f"original Image. GT count={gt_count:.2f}")
    axs[0].axis("off")

    axs[1].imshow(density, cmap="jet")
    axs[1].set_title(f"GT density\ncount≈{density.sum():.2f}")
    axs[1].axis("off")

    # mask
    axs[2].imshow(mask, cmap="gray")
    axs[2].set_title("Mask")
    axs[2].axis("off")

    # overlay
    axs[3].imshow(image)
    # axs[3].imshow(mask, cmap="Reds", alpha=0.35)
    axs[3].set_title(f"Mask Overlay\nGT count={gt_count:.2f}")
    axs[3].axis("off")

    if model:
        model.eval()
        pred_density = model(sample["image"].unsqueeze(0))
        pred_count = pred_density.sum().item() / density_scale
        axs[4].imshow(pred_density.detach().numpy().squeeze(), cmap="jet")
        axs[4].set_title(f"Pred density\ncount={pred_count:.2f}")
        axs[4].axis("off")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train density estimation models.")
    parser.add_argument(
        "--model",
        type=str,
        default="csrnet",
        choices=["unet", "csrnet"],
        help="Which density model to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--count-loss-weight",
        type=float,
        default=0.01,
        help="Weight for the count loss term.",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="shng",
        help="Short identifier for the dataset (used in saved filenames).",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Masking ratio used during training (used in saved filenames).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models",
        help="Directory where trained models and curves are saved.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Number of epochs without improvement to wait before stopping training.",
    )
    parser.add_argument(
        "--density-scale",
        type=float,
        default=DEFAULT_DENSITY_SCALE,
        help="Training only: MSE target = raw_gt * scale; count from pred = pred.sum()/scale. "
        "Must match when visualizing trained checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example: visualize one FSC147 sample using its precomputed density map.
    # The root points to the directory containing annotation_FSC147_384.json.
    # fsc_root = Path("data/FSC147")
    # visualize_fsc147_density(
    #     root=fsc_root,
    #     img_name="1050.jpg",      # pick any id present in Train_Test_Val_FSC_147.json
    #     use_fixed_density=False,  # set True to use density_path_fixed instead
    # )

    # Example of using composed transforms for augmentation:
    # - resize to 512x512
    # - random 90-degree rotation
    # - random horizontal flip
    # - light color jitter
    transform = compose_transforms(
        resize_transform,
        random_90deg_rotation_transform,
        horizontal_flip_transform,
        color_jitter_transform,
    )

    dataset = load_shanghaitech_dataset(
        root="/home/haneenn/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech",
        part=["part_A", "part_B"],
        split="train_data",
        mask_object_ratio=None,
        density_map_dir=DENSITY_MAP_DIR_AUTO,
        keep_original_image=False,
        density_sigma=2.0,
        transform=transform,
    )
    # dataset = load_fsc147_dataset(
    #     root="data/FSC147",
    #     split="train",
    #     density_map_dir="data/FSC147/gt_density_map_adaptive_384_VarV2",
    #     mask_object_ratio=None,
    #     transform=random_crop_transform,
    # )
    # # Pre-compute density maps once (saves under <image_path>/../density_maps/<stem>_density.npy).
    # # After this, __getitem__ loads the saved .npy instead of computing on the fly.
    # precompute_density_maps(dataset)
    print(f"Loaded dataset with {len(dataset)} samples")

    # # # Example: visualize one image and its density (computed on the fly) by image name

    generator = torch.Generator().manual_seed(42)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    # Choose which density model to train based on CLI flag.
    if args.model == "unet":
        model = UNetDensity()
    elif args.model == "csrnet":
        model = CSRNet()
    else:
        raise ValueError(f"Unknown model {args.model!r}")

    train(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        count_loss_weight=args.count_loss_weight,
        model_name=args.model,
        data_name=args.data_name,
        mask_ratio=args.mask_ratio,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping_patience,
        density_scale=args.density_scale,
        batch_size=8,
    )

    # model.load_state_dict(torch.load("best_model.pth"))
    visualize_image_and_density(
        dataset,
        "part_A/train_data/images/IMG_199.jpg",
        use_precomputed_density=True,
        pred_density_scale=args.density_scale,
        # model=model,
    )