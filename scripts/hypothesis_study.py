"""
C1: hypothesis study — does training the model to hallucinate masked-object
density produce a better-counting, more context-aware scene representation?

Trains three variants on the same split and compares them across a grid of
validation mask ratios:

    - ``baseline``: no masking at all (``--mask-ratio None``)
    - ``robust``:   inpainting with ``--mask-mode robust`` (image & density masked)
    - ``inpaint``:  inpainting with ``--mask-mode inpaint`` (image masked, full density)

Each model is then evaluated on the SAME test set across a sweep of val mask ratios
(``0.0, 0.1, 0.3, 0.5, 0.7`` by default) with a fixed seed so holes are identical
across models. The script writes a CSV + a two-panel plot:

    Panel 1: ``MAE_total_masked`` vs ratio for the three models
    Panel 2: ``MAE_hidden``       vs ratio for the three models

The hypothesis is supported if, as the mask ratio grows:
    inpaint.MAE_total_masked < robust.MAE_total_masked < baseline.MAE_total_masked
AND
    inpaint.MAE_hidden stays bounded while baseline/robust diverge.

Usage (train + eval all three):

    python -m scripts.hypothesis_study \
        --epochs 100 --data-part part_A --mask-ratio 0.3

Usage (skip training, reuse checkpoints):

    python -m scripts.hypothesis_study --skip-train \
        --baseline-ckpt trained_models/vit/vit-shng-A-part_A-nomsk/vit-shng-A-nomsk-best.pth \
        --robust-ckpt   trained_models/vit/.../*-best.pth \
        --inpaint-ckpt  trained_models/vit/.../*-best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import DENSITY_MAP_DIR_AUTO
from data.shanghaitech import load_shanghaitech_dataset
from data.transforms import vit_normalize_only_transform
from model.vit_density import ViTDensity
from scripts.gpu_state import get_free_gpu_indices
from training.train import (
    DEFAULT_DENSITY_SCALE,
    validate,
    validate_masked,
)


def _build_model(device: torch.device, linear_probe: bool = False) -> ViTDensity:
    model = ViTDensity(
        encoder_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True,
        freeze_encoder=False,
        output_activation="softplus",
        linear_probe=linear_probe,
    )
    return model.to(device)


def _load_checkpoint(model: ViTDensity, ckpt_path: Path, device: torch.device) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state, strict=True)


def _eval_dataset(
    args: argparse.Namespace,
    data_part: str,
    mask_dot_style: str,
    mask_ratio: float,
    eval_transform,
) -> Any:
    kwargs = dict(
        root=args.data_root,
        part=[data_part],
        split=args.split,
        density_map_dir=DENSITY_MAP_DIR_AUTO,
        keep_original_image=False,
        density_geometry_adaptive=True,
        density_beta=0.3,
        density_k=3,
        density_min_sigma=4.0,
        transform=eval_transform,
        mask_dot_style=mask_dot_style,
        mask_dot_box_aspect=(3, 1),
        mask_dot_sigma_to_box=4.0,
    )
    if mask_ratio > 0:
        kwargs.update(
            mask_object_ratio=float(mask_ratio),
            mask_mode="inpaint",
            deterministic_masks=True,
            mask_seed=0,
        )
    else:
        kwargs.update(mask_object_ratio=None)
    return load_shanghaitech_dataset(**kwargs)


def _run_one_eval(
    model: ViTDensity,
    dataset: Any,
    device: torch.device,
    mask_ratio: float,
    tiled: bool,
    tile: int,
    overlap: int,
    max_batch: int,
    val_batch_size: int,
) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    clean_mae = validate(
        model,
        loader,
        device,
        density_scale=DEFAULT_DENSITY_SCALE,
        gt_downsample="bilinear",
        tiled=tiled,
        tile_size=tile,
        tile_overlap=overlap,
        tile_max_batch=max_batch,
    )
    if mask_ratio > 0:
        m = validate_masked(
            model,
            loader,
            device,
            density_scale=DEFAULT_DENSITY_SCALE,
            gt_downsample="bilinear",
            tiled=tiled,
            tile_size=tile,
            tile_overlap=overlap,
            tile_max_batch=max_batch,
        )
        return {
            "mae_total": m["mae_total_masked"],
            "mae_hidden": m["mae_hidden"],
            "mae_visible": m["mae_visible_masked"],
            "mean_mask_fraction": m["mean_mask_fraction"],
        }
    return {
        "mae_total": float(clean_mae),
        "mae_hidden": 0.0,
        "mae_visible": float(clean_mae),
        "mean_mask_fraction": 0.0,
    }


VariantConfig = Dict[str, Any]


def _argv_to_flag_dict(argv: List[str]) -> Dict[str, Any]:
    """Parse `--flag value` / `--bool-flag` argv into a simple dict."""
    out: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[tok] = argv[i + 1]
            i += 2
        else:
            out[tok] = None
            i += 1
    return out


def _default_variants(
    args: argparse.Namespace, main_passthrough_argv: List[str]
) -> Dict[str, VariantConfig]:
    """Three training configs that share all non-masking hyperparameters."""
    shared_from_main = _argv_to_flag_dict(main_passthrough_argv)
    # These are controlled by this script per-variant (or per-study), not copied
    # from passthrough.
    for k in ("--mask-ratio", "--mask-mode", "--val-mask-ratio"):
        shared_from_main.pop(k, None)

    shared = {
        **shared_from_main,
        "--output-dir": str(args.output_root / "runs"),
    }

    variants: Dict[str, VariantConfig] = {
        "baseline": {
            **shared,
            "--val-mask-ratio": "0",
        },
        "robust": {
            **shared,
            "--mask-ratio": str(args.mask_ratio),
            "--mask-mode": "robust",
            "--val-mask-ratio": str(args.mask_ratio),
        },
        "inpaint": {
            **shared,
            "--mask-ratio": str(args.mask_ratio),
            "--mask-mode": "inpaint",
            "--val-mask-ratio": str(args.mask_ratio),
        },
    }
    return variants


def _flags_to_argv(flags: VariantConfig) -> List[str]:
    argv: List[str] = []
    for k, v in flags.items():
        argv.append(k)
        if v is not None:
            argv.append(str(v))
    return argv


def _run_training(
    variant: str,
    flags: VariantConfig,
    log_path: Path,
    gpu_id: Optional[int] = None,
    main_passthrough_argv: List[str] | None = None,
) -> int:
    """Launch main.py in a subprocess; tee output to log_path."""
    # Allow this script to accept additional main.py flags without having to
    # re-declare every single one here. We place passthrough flags before the
    # generated variant flags so variant-defining options keep precedence.
    passthrough = main_passthrough_argv or []
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        *passthrough,
        *_flags_to_argv(flags),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    gpu_note = ""
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_note = f" (GPU {gpu_id})"
    print(f"[{variant}] launching{gpu_note}: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        if gpu_id is not None:
            f.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
        )
        rc = proc.wait()
    print(f"[{variant}] exit={rc} log={log_path}")
    return rc


def _spawn_training(
    variant: str,
    flags: VariantConfig,
    log_path: Path,
    gpu_id: Optional[int] = None,
    main_passthrough_argv: List[str] | None = None,
) -> tuple[subprocess.Popen, Any]:
    passthrough = main_passthrough_argv or []
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        *passthrough,
        *_flags_to_argv(flags),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    gpu_note = ""
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_note = f" (GPU {gpu_id})"
    print(f"[{variant}] launching async{gpu_note}: {' '.join(cmd)}")
    f = log_path.open("w", encoding="utf-8")
    f.write(f"# cmd: {' '.join(cmd)}\n")
    if gpu_id is not None:
        f.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n")
    f.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    return proc, f


def _find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    matches = list(run_dir.glob("*-best.pth"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _infer_run_dir(flags: VariantConfig, output_root: Path) -> Path:
    """Mirror the run-tag logic in main.py so we can locate the trained model."""
    mask_ratio = flags.get("--mask-ratio")
    mask_mode = flags.get("--mask-mode", "inpaint")
    dot_style = flags.get("--mask-dot-style", "box")
    if mask_ratio is None:
        mask_str = "nomsk"
    else:
        mask_str = f"{mask_ratio}-{mask_mode}"
        if dot_style and dot_style != "box":
            mask_str = f"{mask_str}-{dot_style}"
        mask_str = f"{mask_str}-random"  # main.py default (not --deterministic-masks)
    model_name = flags.get("--model", "vit")
    data_name = flags.get("--data-name", "shng-A")
    data_part = flags.get("--data-part", "part_B")
    run_tag = f"{model_name}-{data_name}-{data_part}-{mask_str}"
    return output_root / "runs" / model_name / run_tag


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    study = p.add_argument_group("Hypothesis-study configuration")

    # Study-orchestration / evaluation options.
    study.add_argument("--output-root", type=Path, default=Path("studies/hypothesis_study"))
    study.add_argument("--timestamped", action="store_true", help="Append a timestamp to --output-root.")
    study.add_argument("--data-root", type=str, default="./ShanghaiTech")
    study.add_argument("--split", type=str, default="test_data")
    study.add_argument("--tile", type=int, default=224)
    study.add_argument("--overlap", type=int, default=48)
    study.add_argument("--tile-max-batch", type=int, default=16)
    study.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7],
                       help="Val mask ratios to evaluate each model on.")
    study.add_argument("--skip-train", action="store_true",
                       help="Don't retrain; use provided / discovered checkpoints.")
    study.add_argument("--baseline-ckpt", type=Path, default=None)
    study.add_argument("--robust-ckpt", type=Path, default=None)
    study.add_argument("--inpaint-ckpt", type=Path, default=None)

    study.add_argument(
        "--mask-ratio",
        type=float,
        default=0.3,
        help="Training mask ratio for robust and inpaint variants; also used for --val-mask-ratio.",
    )

    args, main_passthrough_argv = p.parse_known_args()
    return args, main_passthrough_argv


def main() -> None:
    args, main_passthrough_argv = parse_args()
    main_flags = _argv_to_flag_dict(main_passthrough_argv)
    out_root: Path = args.output_root
    if args.timestamped:
        out_root = out_root / datetime.now().strftime("%Y-%m-%d-%Hh%M")
    out_root.mkdir(parents=True, exist_ok=True)
    args.output_root = out_root
    print(f"Hypothesis study output: {out_root}")

    variants = _default_variants(args, main_passthrough_argv)
    data_part = str(main_flags.get("--data-part", "part_B"))
    mask_dot_style = str(main_flags.get("--mask-dot-style", "box"))
    tiled_val = "--tiled-val" in main_flags
    vit_native_resolution = "--vit-native-resolution" in main_flags

    # -------- training phase --------
    ckpts: Dict[str, Path] = {}
    if not args.skip_train:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this study requires at least 2 free GPUs.")
        free_gpus = get_free_gpu_indices()
        num_free_gpus=len(free_gpus)-1 # one gpu is preserved to avoid crashiung the server
        print(f"Free GPUs detected: {free_gpus} | num_free_gpus: {num_free_gpus}")
        if num_free_gpus == 0:
            raise RuntimeError(
                "This server does not support 4 parallel GPU runs: "
                f"only {num_free_gpus} free GPU(s) detected."
            )

        run_order = ["inpaint", "robust", "baseline"]
        variant_rcs: Dict[str, int] = {}
        variant_logs: Dict[str, Path] = {
            name: out_root / "logs" / f"train-{name}.log" for name in run_order
        }

        if num_free_gpus == 3:
            procs: Dict[str, tuple[subprocess.Popen, Any]] = {}
            for name, gpu_id in zip(run_order, free_gpus[:3]):
                procs[name] = _spawn_training(
                    name,
                    variants[name],
                    variant_logs[name],
                    gpu_id=gpu_id,
                    main_passthrough_argv=main_passthrough_argv,
                )
            for name, (proc, handle) in procs.items():
                rc = proc.wait()
                handle.close()
                variant_rcs[name] = rc
                print(f"[{name}] exit={rc} log={variant_logs[name]}")
        elif num_free_gpus == 2:
            gpu_a, gpu_b = free_gpus[0], free_gpus[1]
            inpaint_proc, inpaint_handle = _spawn_training(
                "inpaint",
                variants["inpaint"],
                variant_logs["inpaint"],
                gpu_id=gpu_a,
                main_passthrough_argv=main_passthrough_argv,
            )
            for name in ("robust", "baseline"):
                rc = _run_training(
                    name,
                    variants[name],
                    variant_logs[name],
                    gpu_id=gpu_b,
                    main_passthrough_argv=main_passthrough_argv,
                )
                variant_rcs[name] = rc
            variant_rcs["inpaint"] = inpaint_proc.wait()
            inpaint_handle.close()
            print(f"[inpaint] exit={variant_rcs['inpaint']} log={variant_logs['inpaint']}")
        elif num_free_gpus == 1:
            gpu_a = free_gpus[0]
            for name in run_order:
                rc = _run_training(
                    name,
                    variants[name],
                    variant_logs[name],
                    gpu_id=gpu_a,
                    main_passthrough_argv=main_passthrough_argv,
                )
                variant_rcs[name] = rc
            variant_rcs["inpaint"] = inpaint_proc.wait()
            inpaint_handle.close()
            print(f"[inpaint] exit={variant_rcs['inpaint']} log={variant_logs['inpaint']}")

        for name in run_order:
            rc = variant_rcs.get(name, 1)
            if rc != 0:
                raise RuntimeError(
                    f"Training variant {name!r} failed (rc={rc}); see {variant_logs[name]}"
                )
            run_dir = _infer_run_dir(variants[name], out_root)
            best = _find_best_checkpoint(run_dir)
            if best is None:
                raise RuntimeError(
                    f"No *-best.pth found in {run_dir}; training may have finished "
                    f"without a best-MAE improvement (check log: {variant_logs[name]})."
                )
            ckpts[name] = best
    else:
        explicit = {
            "baseline": args.baseline_ckpt,
            "robust": args.robust_ckpt,
            "inpaint": args.inpaint_ckpt,
        }
        for name, flags in variants.items():
            if explicit[name] is not None:
                ckpts[name] = explicit[name]
            else:
                run_dir = _infer_run_dir(flags, out_root)
                best = _find_best_checkpoint(run_dir)
                if best is None:
                    raise FileNotFoundError(
                        f"Checkpoint for {name!r} not provided and none found under {run_dir}. "
                        f"Pass --{name}-ckpt."
                    )
                ckpts[name] = best
        print(f"Using checkpoints: {ckpts}")

    # -------- evaluation phase --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_csv = out_root / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "val_mask_ratio",
            "mae_total", "mae_hidden", "mae_visible", "mean_mask_fraction",
            "checkpoint",
        ])

        all_points: Dict[str, List[Dict[str, float]]] = {}
        for name, ckpt in ckpts.items():
            print(f"\n=== Evaluating variant={name}  ckpt={ckpt} ===")
            model = _build_model(device)
            _load_checkpoint(model, ckpt, device)
            eval_transform = vit_normalize_only_transform(model.encoder)

            variant_points: List[Dict[str, float]] = []
            for r in args.ratios:
                dataset = _eval_dataset(
                    args,
                    data_part=data_part,
                    mask_dot_style=mask_dot_style,
                    mask_ratio=float(r),
                    eval_transform=eval_transform,
                )
                metrics = _run_one_eval(
                    model,
                    dataset,
                    device,
                    mask_ratio=float(r),
                    tiled=tiled_val,
                    tile=int(args.tile),
                    overlap=int(args.overlap),
                    max_batch=int(args.tile_max_batch),
                    val_batch_size=1 if vit_native_resolution else 8,
                )
                print(
                    f"  ratio={r:.2f}  mae_total={metrics['mae_total']:.3f}  "
                    f"mae_hidden={metrics['mae_hidden']:.3f}  "
                    f"mae_visible={metrics['mae_visible']:.3f}  "
                    f"mask_frac={metrics['mean_mask_fraction']:.3f}"
                )
                writer.writerow([
                    name, f"{r:.3f}",
                    f"{metrics['mae_total']:.6f}",
                    f"{metrics['mae_hidden']:.6f}",
                    f"{metrics['mae_visible']:.6f}",
                    f"{metrics['mean_mask_fraction']:.6f}",
                    str(ckpt),
                ])
                f.flush()
                variant_points.append({"ratio": float(r), **metrics})
            all_points[name] = variant_points

    print(f"\nResults CSV: {results_csv}")

    # -------- plot --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"baseline": "tab:gray", "robust": "tab:orange", "inpaint": "tab:blue"}
    markers = {"baseline": "o", "robust": "s", "inpaint": "^"}

    for name, points in all_points.items():
        xs = [p["ratio"] for p in points]
        ax = axes[0]
        ax.plot(xs, [p["mae_total"] for p in points],
                color=colors.get(name, None), marker=markers.get(name, "o"),
                label=name)
        ax = axes[1]
        ax.plot(xs, [p["mae_hidden"] for p in points],
                color=colors.get(name, None), marker=markers.get(name, "o"),
                label=name)

    axes[0].set_title("MAE_total vs val mask ratio (on masked input)")
    axes[0].set_xlabel("Val mask ratio")
    axes[0].set_ylabel("Per-image MAE (count)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("MAE_hidden vs val mask ratio (hallucination test)")
    axes[1].set_xlabel("Val mask ratio")
    axes[1].set_ylabel("Per-image MAE on hidden count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out_root / "results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot: {plot_path}")

    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"variants": {k: str(v) for k, v in ckpts.items()},
                   "ratios": list(args.ratios),
                   "results": all_points}, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
