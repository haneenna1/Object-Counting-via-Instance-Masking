import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import visualize_image_and_density
from data.dataset import DENSITY_MAP_DIR_AUTO
from data.shanghaitech import load_shanghaitech_dataset
# from data.transforms import timm_eval_dict_transform
from data.transforms import vit_normalize_only_transform
from model.csrnet import CSRNet, load_vgg16_frontend
from model.unet import UNetDensity
from model.vit_density import ViTDensity
from training.train import DEFAULT_DENSITY_SCALE, downsample_gt_csrnet_cubic, predict_tiled


def _resize_gt_density_to_prediction(
    gt_density: torch.Tensor,
    pred_density: torch.Tensor,
    gt_downsample: str = "bilinear",
) -> torch.Tensor:
    """
    Return GT density in the same spatial resolution/units as prediction.
    """
    if gt_downsample not in ("bilinear", "csrnet_cubic"):
        raise ValueError(f"gt_downsample must be 'bilinear' or 'csrnet_cubic', got {gt_downsample!r}")

    if pred_density.shape[-2:] == gt_density.shape[-2:]:
        return gt_density

    h_pred, w_pred = pred_density.shape[-2:]
    if gt_downsample == "csrnet_cubic":
        return downsample_gt_csrnet_cubic(gt_density, h_pred, w_pred)

    h_gt, w_gt = gt_density.shape[-2:]
    gt_down = F.interpolate(
        gt_density,
        size=(h_pred, w_pred),
        mode="bilinear",
        align_corners=False,
    )
    spatial_scale = (h_gt / h_pred) * (w_gt / w_pred)
    return gt_down * spatial_scale


def _build_eval_output_dir(run_name: str, output_root: str | Path) -> Path:
    out_dir = Path(output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@torch.no_grad()
def evaluate_model_on_dataset(
    model,
    dataset,
    *,
    run_name: str = "eval",
    output_root: str | Path = "inference/results",
    device=None,
    density_scale: float = DEFAULT_DENSITY_SCALE,
    gt_downsample: str = "bilinear",
    batch_size: int = 8,
    num_workers: int = 4,
    num_visualizations: int = 10,
    tiled_eval: bool = False,
    tile_size: int = 224,
    tile_overlap: int = 48,
    tile_max_batch: int = 16,
) -> dict[str, float | int | str | None]:
    """
    Evaluate a model on one dataset and persist outputs under inference/results:
    - MAE on total count
    - MAE on visible-object count
    - MAE on invisible-object count
    - N visualization images
    - metrics.json
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = _build_eval_output_dir(run_name=run_name, output_root=output_root)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    was_training = model.training
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    total_abs_full = 0.0
    total_abs_visible = 0.0
    total_abs_invisible = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)

        if tiled_eval:
            if images.size(0) != 1:
                raise RuntimeError(
                    f"tiled eval requires eval batch size 1, got {images.size(0)}."
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
        gt_for_count = _resize_gt_density_to_prediction(
            gt_density=gt_density,
            pred_density=pred_density,
            gt_downsample=gt_downsample,
        )

        pred_raw = pred_density / density_scale
        pred_full = pred_raw.sum(dim=(1, 2, 3))
        gt_full = gt_for_count.sum(dim=(1, 2, 3))
        total_abs_full += torch.abs(pred_full - gt_full).sum().item()

        if mask is None:
            vis_mask = torch.ones_like(gt_for_count)
            inv_mask = torch.zeros_like(gt_for_count)
        else:
            if mask.shape[-2:] != pred_density.shape[-2:]:
                mask = F.interpolate(mask, size=pred_density.shape[-2:], mode="nearest")
            inv_mask = mask.clamp(0.0, 1.0)
            vis_mask = 1.0 - inv_mask

        pred_visible = (pred_raw * vis_mask).sum(dim=(1, 2, 3))
        gt_visible = (gt_for_count * vis_mask).sum(dim=(1, 2, 3))
        pred_invisible = (pred_raw * inv_mask).sum(dim=(1, 2, 3))
        gt_invisible = (gt_for_count * inv_mask).sum(dim=(1, 2, 3))

        total_abs_visible += torch.abs(pred_visible - gt_visible).sum().item()
        total_abs_invisible += torch.abs(pred_invisible - gt_invisible).sum().item()
        total_samples += images.size(0)

    if total_samples == 0:
        raise RuntimeError("Cannot evaluate on an empty dataset.")

    vis_count = min(max(num_visualizations, 0), len(dataset))
    if vis_count > 0:
        indices = np.linspace(0, len(dataset) - 1, vis_count, dtype=int).tolist()
        for idx in indices:
            visualize_image_and_density(
                dataset,
                index=int(idx),
                use_precomputed_density=True,
                use_dataset_item=True,
                pred_density_scale=density_scale,
                save_dir=vis_dir,
                model=model,
                device=device,
            )

    metrics = {
        "mae_full_count": total_abs_full / total_samples,
        "mae_visible_count": total_abs_visible / total_samples,
        "mae_invisible_count": total_abs_invisible / total_samples,
        "num_samples": total_samples,
        "num_visualizations": vis_count,
        "output_dir": str(output_dir),
        "visualization_dir": str(vis_dir),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics["metrics_path"] = str(metrics_path)

    if was_training:
        model.train()
    return metrics


def _build_model(model_name: str, freeze_encoder: bool) -> torch.nn.Module:
    if model_name == "unet":
        return UNetDensity()
    if model_name == "csrnet":
        model = CSRNet()
        load_vgg16_frontend(model, freeze_frontend=freeze_encoder)
        return model
    if model_name == "vit":
        return ViTDensity(
            encoder_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            freeze_encoder=freeze_encoder,
            output_activation="softplus",
        )
    raise ValueError(f"Unknown model {model_name!r}")


def _build_eval_dataset(args, model) -> tuple[object, str, float]:
    # eval_transform = timm_eval_dict_transform(model.encoder) if args.model == "vit" else None
    eval_transform = vit_normalize_only_transform(model.encoder)
    dataset = load_shanghaitech_dataset(
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
    gt_downsample = "csrnet_cubic" if args.model == "csrnet" else "bilinear"
    density_scale = 1.0 if args.model == "csrnet" else args.density_scale
    return dataset, gt_downsample, density_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone inference/evaluation runner.")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-visualizations", type=int, default=10)
    parser.add_argument("--density-scale", type=float, default=DEFAULT_DENSITY_SCALE)
    parser.add_argument("--tiled-eval", action="store_true")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-overlap", type=int, default=48)
    parser.add_argument("--tile-max-batch", type=int, default=16)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="inference/results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(args.model, freeze_encoder=args.freeze_encoder)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    dataset, gt_downsample, density_scale = _build_eval_dataset(args, model)
    if args.run_name:
        run_name = args.run_name
    else:
        checkpoint_stem = Path(args.checkpoint).stem
        test_data_name = f"{args.part}-{args.split}"
        test_mask_tag = (
            f"testmask-{args.mask_ratio:g}"
            if args.mask_ratio is not None
            else "testmask-none"
        )
        date_tag = datetime.now().strftime("%d-%m-%H")
        run_name = f"{checkpoint_stem}-{test_mask_tag}-{test_data_name}-{date_tag}"

    metrics = evaluate_model_on_dataset(
        model,
        dataset,
        run_name=run_name,
        output_root=args.output_root,
        device=device,
        density_scale=density_scale,
        gt_downsample=gt_downsample,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_visualizations=args.num_visualizations,
        tiled_eval=args.tiled_eval,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_max_batch=args.tile_max_batch,
    )
    print(
        "Evaluation complete | "
        f"Full MAE {metrics['mae_full_count']:.4f} | "
        f"Visible MAE {metrics['mae_visible_count']:.4f} | "
        f"Invisible MAE {metrics['mae_invisible_count']:.4f} | "
        f"Saved at {metrics['output_dir']}"
    )
