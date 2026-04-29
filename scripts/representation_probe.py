"""
C2: representation probe — compare the scene-representation quality of models
trained under different masking regimes by freezing each encoder and fitting
a strict 1x1 linear probe density head on top.

The argument: if inpaint-trained encoders are genuinely more "context-aware",
their frozen features should linear-probe to a lower MAE on clean test images
than baseline / robust encoders at the same compute budget -- a model-agnostic
statement about the representation, independent of the decoder that produced it.

Usage:

    python -m scripts.representation_probe \
        --probe-epochs 80 \
        --baseline-ckpt trained_models/vit/.../*-best.pth \
        --robust-ckpt   trained_models/vit/.../*-best.pth \
        --inpaint-ckpt  trained_models/vit/.../*-best.pth

The script:
  1. For each input checkpoint, spawns ``main.py`` with the flags needed to
     train ONLY a 1x1 linear density head on top of that frozen backbone.
  2. Parses the resulting history.json / log to extract final ``best_mae``.
  3. Writes a CSV and a bar chart comparing the three.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_state import get_free_gpu_indices


def _run_probe(
    label: str,
    backbone_ckpt: Path,
    epochs: int,
    out_dir: Path,
    main_passthrough_argv: List[str],
    gpu_id: Optional[int] = None,
) -> Path:
    """Run main.py with linear-probe + backbone-only init from ``backbone_ckpt``.
    Returns the run directory (where history.json / curves.png live)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO_ROOT / "main.py"),
        # Allow callers to pass arbitrary main.py flags without duplicating
        # that full surface area here; probe-specific flags below keep precedence.
        *main_passthrough_argv,
        "--model", "vit",
        "--linear-probe",
        "--freeze-encoder",
        "--vit-init-checkpoint", str(backbone_ckpt),
        "--vit-init-load-mode", "backbone",
        "--epochs", str(epochs),
        "--output-dir", str(out_dir),
        "--data-name", f"probe-{label}",
    ]
    log_path = out_dir / f"probe-{label}.log"
    env = os.environ.copy()
    gpu_note = ""
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_note = f" (GPU {gpu_id})"
    print(f"[{label}] launching{gpu_note}: {' '.join(cmd)}")
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
    if rc != 0:
        raise RuntimeError(f"Probe for {label!r} failed (rc={rc}); see {log_path}")
    return out_dir


def _spawn_probe(
    label: str,
    backbone_ckpt: Path,
    epochs: int,
    out_dir: Path,
    main_passthrough_argv: List[str],
    gpu_id: Optional[int] = None,
) -> Tuple[subprocess.Popen, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO_ROOT / "main.py"),
        *main_passthrough_argv,
        "--model", "vit",
        "--linear-probe",
        "--freeze-encoder",
        "--vit-init-checkpoint", str(backbone_ckpt),
        "--vit-init-load-mode", "backbone",
        "--epochs", str(epochs),
        "--output-dir", str(out_dir),
        "--data-name", f"probe-{label}",
    ]
    log_path = out_dir / f"probe-{label}.log"
    env = os.environ.copy()
    gpu_note = ""
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_note = f" (GPU {gpu_id})"
    print(f"[{label}] launching async{gpu_note}: {' '.join(cmd)}")
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


def _find_history_json(out_dir: Path, label: str) -> Optional[Path]:
    """Locate the history.json written by train.train(). Search under the run dir."""
    candidates = list(out_dir.rglob(f"*probe-{label}*-history.json"))
    candidates += list(out_dir.rglob("*-history.json"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _final_val_mae_from_history(history_path: Path) -> Dict[str, float]:
    """Return ``{"best_val_mae": ..., "final_val_mae": ..., "epochs": N}``."""
    with history_path.open("r", encoding="utf-8") as f:
        h = json.load(f)
    vals = h.get("val_mae", []) or []
    if not vals:
        return {"best_val_mae": float("nan"), "final_val_mae": float("nan"), "epochs": 0}
    return {
        "best_val_mae": float(min(vals)),
        "final_val_mae": float(vals[-1]),
        "epochs": len(vals),
        "best_val_mae_hidden": float(min(h.get("val_mae_hidden", [float("nan")]))) if h.get("val_mae_hidden") else float("nan"),
    }


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=Path("studies/representation_probe"))
    p.add_argument("--timestamped", action="store_true")
    p.add_argument("--probe-epochs", type=int, default=80)

    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--robust-ckpt", type=Path, default=None)
    p.add_argument("--inpaint-ckpt", type=Path, default=None)
    p.add_argument(
        "--extra-ckpt",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional additional <label>=<checkpoint> pairs, can be repeated.",
    )
    args, main_passthrough_argv = p.parse_known_args()
    return args, main_passthrough_argv


def main() -> None:
    args, main_passthrough_argv = parse_args()
    out_root: Path = args.output_root
    if args.timestamped:
        out_root = out_root / datetime.now().strftime("%Y-%m-%d-%Hh%M")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Representation probe output: {out_root}")

    probes: Dict[str, Path] = {}
    if args.baseline_ckpt is not None:
        probes["baseline"] = args.baseline_ckpt
    if args.robust_ckpt is not None:
        probes["robust"] = args.robust_ckpt
    if args.inpaint_ckpt is not None:
        probes["inpaint"] = args.inpaint_ckpt
    for raw in args.extra_ckpt:
        if "=" not in raw:
            raise ValueError(f"--extra-ckpt expects LABEL=PATH, got {raw!r}")
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in --extra-ckpt: {raw!r}")
        probes[label] = Path(path)

    if not probes:
        raise SystemExit("Nothing to probe: pass at least one of "
                         "--baseline-ckpt / --robust-ckpt / --inpaint-ckpt / --extra-ckpt.")

    for label, ckpt in probes.items():
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint for {label!r} not found: {ckpt}")

    free_gpus = get_free_gpu_indices()
    num_free_gpus = len(free_gpus) - 1  # one GPU is preserved to avoid crashing the server
    print(f"Free GPUs detected: {free_gpus} | num_free_gpus: {num_free_gpus}")
    if num_free_gpus == 0:
        raise RuntimeError(
            "This server does not support 4 parallel GPU runs: "
            f"only {num_free_gpus} free GPU(s) detected."
        )

    run_order = list(probes.keys())
    probe_rcs: Dict[str, int] = {}
    run_dirs: Dict[str, Path] = {label: out_root / f"probe-{label}" for label in run_order}

    if num_free_gpus == 3:
        procs: Dict[str, Tuple[subprocess.Popen, Any]] = {}
        for label, gpu_id in zip(run_order, free_gpus[:3]):
            procs[label] = _spawn_probe(
                label,
                probes[label],
                epochs=args.probe_epochs,
                out_dir=run_dirs[label],
                main_passthrough_argv=main_passthrough_argv,
                gpu_id=gpu_id,
            )
        for label, (proc, handle) in procs.items():
            rc = proc.wait()
            handle.close()
            probe_rcs[label] = rc
            print(f"[{label}] exit={rc} run_dir={run_dirs[label]}")
    elif num_free_gpus == 2:
        gpu_a, gpu_b = free_gpus[0], free_gpus[1]
        first = run_order[0]
        async_proc, async_handle = _spawn_probe(
            first,
            probes[first],
            epochs=args.probe_epochs,
            out_dir=run_dirs[first],
            main_passthrough_argv=main_passthrough_argv,
            gpu_id=gpu_a,
        )
        for label in run_order[1:]:
            _run_probe(
                label,
                probes[label],
                epochs=args.probe_epochs,
                out_dir=run_dirs[label],
                main_passthrough_argv=main_passthrough_argv,
                gpu_id=gpu_b,
            )
            probe_rcs[label] = 0
        async_rc = async_proc.wait()
        async_handle.close()
        probe_rcs[first] = async_rc
        print(f"[{first}] exit={probe_rcs[first]} run_dir={run_dirs[first]}")
    elif num_free_gpus == 1:
        gpu_a = free_gpus[0]
        for label in run_order:
            _run_probe(
                label,
                probes[label],
                epochs=args.probe_epochs,
                out_dir=run_dirs[label],
                main_passthrough_argv=main_passthrough_argv,
                gpu_id=gpu_a,
            )
            probe_rcs[label] = 0

    results_csv = out_root / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "checkpoint", "probe_run_dir",
            "epochs_run", "best_val_mae", "final_val_mae", "best_val_mae_hidden",
        ])
        summary: Dict[str, Dict[str, float]] = {}
        for label, ckpt in probes.items():
            rc = probe_rcs.get(label, 1)
            if rc != 0:
                raise RuntimeError(
                    f"Probe for {label!r} failed (rc={rc}); see {run_dirs[label] / f'probe-{label}.log'}"
                )
            run_out = run_dirs[label]
            history = _find_history_json(run_out, label)
            if history is None:
                raise RuntimeError(f"Could not find history.json under {run_out}")
            stats = _final_val_mae_from_history(history)
            summary[label] = stats
            writer.writerow([
                label, str(ckpt), str(run_out),
                stats.get("epochs", 0),
                f"{stats.get('best_val_mae', float('nan')):.6f}",
                f"{stats.get('final_val_mae', float('nan')):.6f}",
                f"{stats.get('best_val_mae_hidden', float('nan')):.6f}",
            ])
            f.flush()
            print(f"[{label}] best={stats['best_val_mae']:.3f}  "
                  f"final={stats['final_val_mae']:.3f}  "
                  f"hidden={stats.get('best_val_mae_hidden', float('nan')):.3f}")

    print(f"\nResults CSV: {results_csv}")

    labels = list(summary.keys())
    best_vals = [summary[l]["best_val_mae"] for l in labels]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, best_vals, color=["tab:gray", "tab:orange", "tab:blue"][:len(labels)])
    for bar, v in zip(bars, best_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Linear-probe best val MAE (lower = better representation)")
    plt.title("Frozen-encoder linear probe: representation quality")
    plt.tight_layout()
    plot_path = out_root / "results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot: {plot_path}")

    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "checkpoints": {k: str(v) for k, v in probes.items()},
            "probe_epochs": args.probe_epochs,
            "results": summary,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
