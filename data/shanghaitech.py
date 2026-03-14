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

import random

import torch
import torch.nn.functional as F

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - runtime environment dependent
    loadmat = None  # type: ignore[assignment]

from data.annotation_types import AnnotationType
from data.dataset import ObjectCountingDataset


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
    part: str = "part_A_final",
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
    """
    root = Path(root)
    images_dir = root / part / split / "images"
    gt_dir = root / part / split / "ground-truth"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    samples: List[Dict[str, Any]] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. 'IMG_1'
        mat_path = gt_dir / f"GT_{stem}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing GT file for {img_path.name}: {mat_path}")

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
        part: str = "part_A_final",
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



def resize_transform(sample, size=(512, 512)):
    image = sample["image"]
    _,h,w = image.shape
    new_h, new_w = size
    density = sample["density"]
    mask = sample["mask"]

    image = F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    density = F.interpolate(
        density.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    scale = (new_h * new_w) / (h * w)
    density = density / scale

    mask = F.interpolate(
        mask.unsqueeze(0),
        size=size,
        mode="nearest"
    ).squeeze(0)

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return sample

def random_crop_transform(sample, crop_size=(512, 512)):
    image = sample["image"]
    density = sample["density"]
    mask = sample["mask"]



    _, H, W = image.shape
    crop_h, crop_w = crop_size

    # ---- pad if needed ----
    pad_h = max(crop_h - H, 0)
    pad_w = max(crop_w - W, 0)

    b_im_sh= image.shape
    b_d_sh=  density.shape
    b_msk_sh=  mask.shape

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h))
        density = F.pad(density, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))

        H = H + pad_h
        W = W + pad_w

    # ---- random crop ----
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    image = image[:, top:top+crop_h, left:left+crop_w]
    density = density[:, top:top+crop_h, left:left+crop_w]
    mask = mask[:, top:top+crop_h, left:left+crop_w]
    
    if image.shape!=(3, 512, 512)  or mask.shape !=(1,512,512) or density.shape !=(1,512,512):
        print("before image shape:", b_im_sh, "after image shape:", image.shape)
        print("before density shape:", b_d_sh, "after density shape:", density.shape)
        print("before mask shape:", b_msk_sh, "after mask shape:", mask.shape)

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return sample



def load_shanghaitech_dataset(
    root: Path | str,
    part: str = "part_A_final",
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
        transform=random_crop_transform,
        **kwargs,
    )

