"""
ShanghaiTech dataset helpers for the `data` module.

This module shows how to:
- Build `samples` for ObjectCountingDataset from ShanghaiTech's images + .mat annotations
- Instantiate an ObjectCountingDataset (and optionally TorchObjectCountingDataset)

Expected directory layout:

    root/
      part_A_final/
        train_data/
          images/       IMG_1.jpg, IMG_2.jpg, ...
          ground_truth/ GT_IMG_1.mat, GT_IMG_2.mat, ...
        test_data/
          images/
          ground_truth/
      part_B_final/
        ...
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - runtime environment dependent
    loadmat = None  # type: ignore[assignment]

from data.annotation_types import AnnotationType
from data.dataset import ObjectCountingDataset
from data.transforms import random_crop_transform, resize_transform


def _require_scipy() -> None:
    if loadmat is None:
        raise ImportError(
            "scipy is required to load ShanghaiTech .mat files. "
            "Install it in your environment (e.g. `conda install scipy`)."
        )


def load_points_from_mat(mat_path: Path) -> List[Tuple[float, float]]:
    """
    Load head point coordinates from a ShanghaiTech .mat file.

    Returns a list of (x, y) pixel coordinates.
    """
    _require_scipy()
    data = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)  # type: ignore[arg-type]
    pts = None

    # Common case: 'annPoints' key with shape (N, 2)
    if data["image_info"].location.any():
        pts = np.asarray(data["image_info"].location)
    else:
        raise RuntimeError(
            f"Could not find (N,2) point array in {mat_path}. "
            "Expected 'location' or similar."
        )

    return [(float(x), float(y)) for x, y in pts]


def build_shanghaitech_samples(
    root: Path | str,
    part: Path | str | List[Path | str] = "part_A",
    split: str = "train_data",
) -> List[Dict[str, Any]]:
    """
    Build the `samples` list for ObjectCountingDataset from ShanghaiTech.

    Args:
        root: Path to ShanghaiTech root (contains part_A_final, part_B_final).
        part: e.g. 'part_A_final' or 'part_B_final'.
        split: 'train_data' or 'test_data'.

    Returns:
        List[dict] with keys: image_path (relative to root), annotation_type, annotations.
        If multiple parts are provided, samples from all parts are concatenated.
    """
    root = Path(root)

    # Allow a single part (str/Path) or a list of parts.
    if isinstance(part, (str, Path)):
        parts: List[Path] = [Path(part)]
    else:
        parts = [Path(p) for p in part]

    samples: List[Dict[str, Any]] = []

    for part_dir in parts:
        images_dir = root / part_dir / split / "images"
        gt_dir = root / part_dir / split / "ground-truth"

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

        for img_path in sorted(images_dir.glob("*.jpg")):
            stem = img_path.stem  # e.g. 'IMG_1'
            mat_path = gt_dir / f"GT_{stem}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(
                    f"Missing GT file for {img_path.name}: {mat_path}"
                )

            points = load_points_from_mat(mat_path)
            samples.append(
                {
                    "image_path": str(img_path.relative_to(root)),
                    "annotation_type": AnnotationType.DOT,
                    "annotations": points,
                }
            )

    return samples


class ShanghaiTechDataset(ObjectCountingDataset):
    """
    ObjectCountingDataset subclass specialized for ShanghaiTech.

    It:
      - Discovers images and .mat files under the usual ShanghaiTech layout
      - Parses head point annotations
      - Configures density + instance masking via ObjectCountingDataset
    """

    def __init__(
        self,
        root: Path | str,
        part: Path | str | List[Path | str] = "part_A_final",
        split: str = "train_data",
        *,
        density_sigma: float = 4.0,
        mask_dot_box_size: int | None = 32,
        mask_object_ratio: float | None = 0.5,
        **kwargs: Any,
    ) -> None:
        root_path = Path(root)
        samples = build_shanghaitech_samples(root_path, part=part, split=split)
        super().__init__(
            samples,
            root=root_path,
            density_sigma=density_sigma,
            mask_dot_box_size=mask_dot_box_size,
            mask_object_ratio=mask_object_ratio,
            **kwargs,
        )


def load_shanghaitech_dataset(
    root: Path | str,
    part: Path | str | List[Path | str] = "part_A_final",
    split: str = "train_data",
    *,
    density_sigma: float = 4.0,
    mask_dot_box_size: int | None = 32,
    mask_object_ratio: float | None = 0.5,
    **kwargs: Any,
) -> ObjectCountingDataset:
    """
    Convenience helper: create an ObjectCountingDataset for ShanghaiTech.

    Args:
        root: Path to ShanghaiTech root.
        part: 'part_A_final' or 'part_B_final'.
        split: 'train_data' or 'test_data'.
        density_sigma: Gaussian sigma for dot density generation.
        mask_dot_box_size: Box size (pixels) for masking around each dot.
        mask_object_ratio: Fraction of objects to mask per image (0..1), or None for no masking.
        **kwargs: Passed through to ObjectCountingDataset (e.g. transform).

    Returns:
        ShanghaiTechDataset configured for this part/split.
    """
    return ShanghaiTechDataset(
        root,
        part=part,
        split=split,
        density_sigma=density_sigma,
        mask_dot_box_size=mask_dot_box_size,
        mask_object_ratio=mask_object_ratio,
        # transform=random_crop_transform,
        **kwargs,
    )

