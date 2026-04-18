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

try:
    import umap
except ImportError as exc:
    raise SystemExit(
        "This script requires umap-learn. Install with:\n  pip install umap-learn\n"
    ) from exc

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
    parser = argparse.ArgumentParser(description="Run UMAP on encoder latent vectors.")
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
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP local neighborhood size (must be < num samples).")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP minimum distance between embedded points.")
    parser.add_argument("--n-components", type=int, default=2, help="UMAP embedding dimensionality (2 for a scatter plot).")
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for UMAP (e.g. euclidean, cosine, manhattan).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score latents before UMAP.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="latent_studies/umap_results")
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
        raise RuntimeError("Need at least 2 samples for UMAP.")
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be >= 2.")
    if args.n_neighbors >= n_samples:
        raise ValueError(
            f"--n-neighbors ({args.n_neighbors}) must be smaller than number of samples ({n_samples}). "
            "Try lowering --n-neighbors or increasing --max-samples."
        )
    if args.n_components < 1:
        raise ValueError("--n-components must be >= 1.")

    x = latent_matrix.astype(np.float64, copy=False)
    if args.standardize:
        x = StandardScaler().fit_transform(x)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=args.seed,
        verbose=False,
    )
    embedding = reducer.fit_transform(x)

    out_dir = make_output_dir(args)
    np.savez_compressed(
        out_dir / "umap_data.npz",
        embedding=embedding.astype(np.float32),
        latent_matrix=latent_matrix,
        counts=count_values,
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
        plt.title(f"UMAP of {args.model} encoder latent space")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(out_dir / "umap_scatter_by_count.png", dpi=180)
        plt.close()
        plot_path = str(out_dir / "umap_scatter_by_count.png")
    else:
        plot_path = ""

    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "num_samples": int(n_samples),
        "latent_dim": int(latent_dim),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "n_components": int(args.n_components),
        "metric": str(args.metric),
        "standardize": bool(args.standardize),
        "seed": int(args.seed),
        "output_dir": str(out_dir),
        "umap_npz": str(out_dir / "umap_data.npz"),
        "plot_path": plot_path,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved UMAP outputs to {out_dir}")


if __name__ == "__main__":
    main()
