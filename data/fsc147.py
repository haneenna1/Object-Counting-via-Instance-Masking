"""
FSC147 dataset helpers for the `data` module, adapted to the
`data/FSC147` directory in this project.

Expected directory layout:

    root/
      annotation_FSC147_384.json
      Train_Test_Val_FSC_147.json
      ImageClasses_FSC147.txt

The annotation JSON has entries like:

    "1050.jpg": {
        "H": 1065,
        "W": 1300,
        "box_examples_coordinates": [...],
        "box_examples_path": [...],
        "density_path": ".../gt_density_map_adaptive_384_VarV2/1050.npy",
        "density_path_fixed": ".../gt_density_map_fixed/1050.npy",
        "img_path": ".../images_384_VarV2/1050.jpg",
        "points": [...]
        ...
    }

We use:
- `points` as dot annotations for object counting supervision
- `img_path` as the (typically absolute) image path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt

from data.annotation_types import AnnotationType
from data.dataset import ObjectCountingDataset


def _load_fsc147_annotation_index(
    root: Path,
    annotation_filename: str = "annotation_FSC147_384.json",
) -> Dict[str, Any]:
    """
    Load the FSC147 annotation index JSON from the given root directory.

    Returns:
        Dict mapping image filename (e.g. "1050.jpg") to its annotation dict.
    """
    annotation_path = root / annotation_filename
    if not annotation_path.is_file():
        raise FileNotFoundError(f"FSC147 annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_fsc147_split(
    root: Path,
    split: str,
    split_filename: str = "Train_Test_Val_FSC_147.json",
) -> Iterable[str]:
    """
    Load image ids for a given split ("train", "val", "test") from the split JSON.
    """
    split_path = root / split_filename
    if not split_path.is_file():
        raise FileNotFoundError(f"FSC147 split file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split not in split_data:
        raise KeyError(
            f"Split '{split}' not found in {split_path}. "
            f"Available keys: {list(split_data.keys())}"
        )

    return split_data[split]


def build_fsc147_samples(
    root: Path | str,
    split: str = "train",
    *,
    annotation_filename: str = "annotation_FSC147_384.json",
    split_filename: str = "Train_Test_Val_FSC_147.json",
) -> List[Dict[str, Any]]:
    """
    Build the `samples` list for ObjectCountingDataset from FSC147.

    Args:
        root: Path to FSC147 root (the `data/FSC147` directory).
        split: One of "train", "val", or "test".
        annotation_filename: Name of the main annotation JSON file.
        split_filename: Name of the JSON file defining train/val/test splits.

    Returns:
        List[dict] with keys:
            - "image_path": image file path (taken directly from "img_path")
            - "annotation_type": AnnotationType.DOT
            - "annotations": list of (x, y) dot coordinates
    """
    root_path = Path(root)
    annotations_index = _load_fsc147_annotation_index(root_path, annotation_filename)
    split_ids: Iterable[str] = _load_fsc147_split(root_path, split, split_filename)

    samples: List[Dict[str, Any]] = []
    for img_name in split_ids:
        if img_name not in annotations_index:
            # Skip ids that do not have annotations.
            continue

        img_info = annotations_index[img_name]
        points_raw = img_info.get("points", [])

        points: List[Tuple[float, float]] = [
            (float(x), float(y)) for x, y in points_raw
        ]

        # Construct image path from the local FSC147 images directory instead of
        # relying on the "img_path" entry from the JSON.
        img_path_str = str(root_path / "images_384_VarV2" / img_name)

        samples.append(
            {
                # Full path under data/FSC147/images_384_VarV2; we pass root=None
                # to ObjectCountingDataset so this is treated as fully-qualified.
                "image_path": img_path_str,
                "annotation_type": AnnotationType.DOT,
                "annotations": points,
            }
        )

    return samples


class FSC147Dataset(ObjectCountingDataset):
    """
    ObjectCountingDataset subclass specialized for FSC147.

    It:
      - Reads annotation JSONs from the `data/FSC147`-style directory
      - Uses dot annotations from the "points" field
      - Uses image paths directly from "img_path" in the JSON
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        *,
        density_sigma: float = 4.0,
        mask_dot_box_size: int | None = None,
        mask_object_ratio: float | None = 0.5,
        **kwargs: Any,
    ) -> None:
        # root is only used to locate the annotation JSON files; image paths
        # in the JSON are already absolute (or otherwise fully-qualified).
        samples = build_fsc147_samples(
            root,
            split=split,
        )
        super().__init__(
            samples,
            root=None,  # so image_path is treated as an absolute/fully-qualified path
            density_sigma=density_sigma,
            mask_dot_box_size=mask_dot_box_size,
            mask_object_ratio=mask_object_ratio,
            **kwargs,
        )


def load_fsc147_dataset(
    root: Path | str,
    split: str = "train",
    *,
    density_sigma: float = 4.0,
    mask_dot_box_size: int | None = None,
    mask_object_ratio: float | None = 0.5,
    **kwargs: Any,
) -> ObjectCountingDataset:
    """
    Convenience helper: create an ObjectCountingDataset for FSC147.

    Args:
        root: Path to FSC147 root (the `data/FSC147` directory).
        split: One of "train", "val", or "test".
        density_sigma: Gaussian sigma for dot density generation.
        mask_dot_box_size: Fixed box side length (pixels) per dot, or None for
            sigma-based size (see ObjectCountingDataset / masking).
        mask_object_ratio: Fraction of objects to mask per image (0..1), or
            None for no masking.
        **kwargs: Passed through to ObjectCountingDataset (e.g. transform).

    Returns:
        FSC147Dataset configured for this split.
    """
    return FSC147Dataset(
        root,
        split=split,
        density_sigma=density_sigma,
        mask_dot_box_size=mask_dot_box_size,
        mask_object_ratio=mask_object_ratio,
        **kwargs,
    )


def load_fsc147_density_sample(
    root: Path | str,
    img_name: str,
    *,
    use_fixed_density: bool = False,
    annotation_filename: str = "annotation_FSC147_384.json",
) -> Dict[str, Any]:
    """
    Load a single FSC147 sample using ONLY the precomputed density map.

    Args:
        root: Path to FSC147 root (the `data/FSC147` directory).
        img_name: Image filename key in the annotation JSON, e.g. "1050.jpg".
        use_fixed_density: If True, use "density_path_fixed"; otherwise "density_path".
        annotation_filename: Name of the annotation JSON file.

    Returns:
        Dict with:
            - "image": tensor (3,H,W) in [0,1]
            - "density": numpy array (H,W) from the .npy file
            - "density_path": path to the .npy file
            - "img_path": path to the image file
    """
    root_path = Path(root)
    anno = _load_fsc147_annotation_index(root_path, annotation_filename)
    if img_name not in anno:
        raise KeyError(f"Image '{img_name}' not found in {annotation_filename}")

    info = anno[img_name]

    # Image path: stored under images_384_VarV2 in this project.
    stem = Path(img_name).stem
    img_path = root_path / "images_384_VarV2" / f"{stem}.jpg"

    # Density path: for this project, density maps live under
    # data/FSC147/gt_density_map_adaptive_384_VarV2 (and a fixed variant).
    if use_fixed_density:
        density_dir = root_path / "gt_density_map_fixed"
    else:
        density_dir = root_path / "gt_density_map_adaptive_384_VarV2"
    density_path = density_dir / f"{stem}.npy"

    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    if not density_path.is_file():
        raise FileNotFoundError(f"Density file not found: {density_path}")

    image = read_image(str(img_path)).float() / 255.0
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    density = np.load(str(density_path)).astype(np.float32)

    return {
        "image": image,
        "density": density,
        "density_path": density_path,
        "img_path": img_path,
    }


def visualize_fsc147_density(
    root: Path | str,
    img_name: str,
    *,
    use_fixed_density: bool = False,
    cmap: str = "jet",
) -> None:
    """
    Visualize an FSC147 image with its precomputed density map.

    Shows a side-by-side matplotlib figure with:
    - the RGB image
    - the density heatmap
    and annotates:
    - GT object count from dot annotations
    - Count from summing the density map.

    Args:
        root: Path to FSC147 root (the `data/FSC147` directory).
        img_name: Image filename key in the annotation JSON, e.g. "1050.jpg".
        use_fixed_density: If True, use the fixed-density directory; otherwise adaptive.
        cmap: Matplotlib colormap for the density heatmap.
    """
    sample = load_fsc147_density_sample(
        root,
        img_name,
        use_fixed_density=use_fixed_density,
    )

    image = sample["image"]
    density = sample["density"]

    # Ground-truth count from dot annotations
    root_path = Path(root)
    anno = _load_fsc147_annotation_index(root_path)
    info = anno.get(img_name)
    gt_count = len(info.get("points", [])) if info is not None else None

    # Count from density map (sum of all pixels)
    density_count = float(density.sum())

    # Convert image tensor (3,H,W) to numpy (H,W,3) for plotting
    img_np = image.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_np)
    if gt_count is not None:
        axes[0].set_title(f"Image: {img_name}\nGT count (dots) = {gt_count}")
    else:
        axes[0].set_title(f"Image: {img_name}")
    axes[0].axis("off")

    im = axes[1].imshow(density, cmap=cmap)
    axes[1].set_title(f"Density map\nSum(density) = {density_count:.2f}")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()