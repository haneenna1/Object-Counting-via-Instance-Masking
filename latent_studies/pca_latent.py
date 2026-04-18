import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DENSITY_MAP_DIR_AUTO
from data.shanghaitech import load_shanghaitech_dataset
from data.transforms import timm_eval_dict_transform
from inference.evaluate import _build_model


@torch.no_grad()
def extract_latent_batch(model: torch.nn.Module, images: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    Return one latent vector per image: shape (B, D).
    """
    if model_name == "vit":
        tokens = model.encoder.forward_features(images)
        tokens = tokens[:, model.num_prefix_tokens :, :]
        return tokens.mean(dim=1)

    if model_name == "unet":
        e1 = model.enc1(images)
        e2 = model.enc2(model.pool(e1))
        e3 = model.enc3(model.pool(e2))
        e4 = model.enc4(model.pool(e3))
        b = model.bottleneck(model.pool(e4))
        return F.adaptive_avg_pool2d(b, output_size=1).flatten(1)

    if model_name == "csrnet":
        x = model.conv1_1(images)
        x = model.conv1_2(x)
        x = model.pool1(x)
        x = model.conv2_1(x)
        x = model.conv2_2(x)
        x = model.pool2(x)
        x = model.conv3_1(x)
        x = model.conv3_2(x)
        x = model.conv3_3(x)
        x = model.pool3(x)
        x = model.conv4_1(x)
        x = model.conv4_2(x)
        x = model.conv4_3(x)
        return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)

    raise ValueError(f"Unsupported model name: {model_name}")


def build_dataset(args: argparse.Namespace, model: torch.nn.Module):
    eval_transform = timm_eval_dict_transform(model.encoder) if args.model == "vit" else None
    return load_shanghaitech_dataset(
        root=args.data_root,
        part=[args.part],
        split=args.split,
        density_map_dir=DENSITY_MAP_DIR_AUTO,
        keep_original_image=False,
        density_geometry_adaptive=True,
        density_beta=0.3,
        density_k=3,
        density_min_sigma=4.0,
        mask_object_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_dot_style=args.mask_dot_style,
        deterministic_masks=args.deterministic_masks,
        mask_seed=args.mask_seed,
        transform=eval_transform,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA on encoder latent vectors.")
    parser.add_argument("--model", type=str, default="vit", choices=["unet", "csrnet", "vit"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pth).")
    parser.add_argument("--data-root", type=str, default="./ShanghaiTech")
    parser.add_argument("--part", type=str, default="part_B")
    parser.add_argument("--split", type=str, default="test_data", choices=["train_data", "test_data"])
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-mode", type=str, default="inpaint", choices=["robust", "inpaint"])
    parser.add_argument("--mask-dot-style", type=str, default="box", choices=["box", "gaussian"])
    parser.add_argument("--deterministic-masks", action="store_true")
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--n-components", type=int, default=2, help="Number of principal components (2 for a scatter plot).")
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score latents before PCA (recommended when dimensions have different scales).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="latent_studies/pca_results")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        checkpoint_stem = Path(args.checkpoint).stem
        date_tag = datetime.now().strftime("%d-%m-%H")
        run_name = f"{checkpoint_stem}-{args.model}-{args.part}-{args.split}-{date_tag}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args.model, freeze_encoder=args.freeze_encoder)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    dataset = build_dataset(args, model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    latents = []
    counts = []
    processed = 0
    pbar = tqdm(loader, desc="Extracting latents")
    for batch in pbar:
        images = batch["image"].to(device)
        batch_latent = extract_latent_batch(model, images, args.model).detach().cpu().numpy()
        batch_count = batch["count"].detach().cpu().numpy()
        latents.append(batch_latent)
        counts.append(batch_count)

        processed += images.size(0)
        if args.max_samples > 0 and processed >= args.max_samples:
            break

    latent_matrix = np.concatenate(latents, axis=0)
    count_values = np.concatenate(counts, axis=0).astype(np.float32)
    if args.max_samples > 0:
        latent_matrix = latent_matrix[: args.max_samples]
        count_values = count_values[: args.max_samples]

    n_samples, latent_dim = latent_matrix.shape
    if n_samples < 2:
        raise RuntimeError("Need at least 2 samples for PCA.")
    if args.n_components < 1:
        raise ValueError("--n-components must be >= 1.")
    if args.n_components > min(n_samples, latent_dim):
        raise ValueError(
            f"--n-components ({args.n_components}) cannot exceed min(n_samples, latent_dim) "
            f"({min(n_samples, latent_dim)})."
        )

    x = latent_matrix.astype(np.float64, copy=False)
    if args.standardize:
        x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=args.n_components, svd_solver="full", random_state=args.seed)
    embedding = pca.fit_transform(x)

    out_dir = make_output_dir(args)
    np.savez_compressed(
        out_dir / "pca_data.npz",
        embedding=embedding,
        latent_matrix=latent_matrix,
        counts=count_values,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        mean_=pca.mean_.astype(np.float64),
        components_=pca.components_.astype(np.float64),
    )

    if args.n_components >= 2:
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=count_values,
            cmap="viridis",
            s=16,
            alpha=0.85,
            edgecolors="none",
        )
        plt.colorbar(scatter, label="GT count")
        ev1, ev2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]
        plt.title(f"PCA of {args.model} encoder latent space")
        plt.xlabel(f"PC1 ({ev1 * 100:.1f}% var.)")
        plt.ylabel(f"PC2 ({ev2 * 100:.1f}% var.)")
        plt.tight_layout()
        plt.savefig(out_dir / "pca_scatter_by_count.png", dpi=180)
        plt.close()
        plot_path = str(out_dir / "pca_scatter_by_count.png")
    else:
        plot_path = ""

    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "num_samples": int(n_samples),
        "latent_dim": int(latent_dim),
        "n_components": int(args.n_components),
        "standardize": bool(args.standardize),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "output_dir": str(out_dir),
        "pca_npz": str(out_dir / "pca_data.npz"),
        "plot_path": plot_path,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved PCA outputs to {out_dir}")


if __name__ == "__main__":
    main()
