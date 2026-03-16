"""
Unified object counting dataset: loads images and produces density maps and instance masks
according to annotation type (dot, bbox, segmentation).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from .annotation_types import AnnotationType
from .density import generate_density
from .masking import generate_instance_mask


# Sentinel: use per-image default location <image_path>/../density_maps/<stem>_density.npy
DENSITY_MAP_DIR_AUTO = "auto"

# Scale density from [0, 1] to [0, DENSITY_SCALE] so the network sees larger targets and
# gets stronger gradients. Counts are recovered as density.sum() / DENSITY_SCALE.
# Training must use this same constant when converting density sums to counts.
DENSITY_SCALE = 255.0


# Type aliases for annotations
DotAnnotations = List[Tuple[float, float]]
BboxAnnotations = List[Tuple[float, float, float, float]]
SegmentationAnnotations = List[np.ndarray]



def _load_image(path: Union[str, Path]) -> torch.Tensor:
    """
    Load image as tensor (3,H,W) in [0,1].
    """
    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 1:  # grayscale
        img = img.repeat(3, 1, 1)
    return img

def _parse_annotation_type(value: Any) -> AnnotationType:
    if isinstance(value, AnnotationType):
        return value
    if isinstance(value, str):
        return AnnotationType(value.lower())
    raise ValueError(f"Cannot parse annotation type: {value}")


def _density_map_path_for_sample(
    item: Dict[str, Any],
    root: Optional[Path],
    density_map_dir: Union[Path, str],
) -> Path:
    """
    Path to the saved density map .npy file for a sample.
    If density_map_dir is DENSITY_MAP_DIR_AUTO, uses <image_path>/../density_maps/<stem>_density.npy.
    Otherwise uses density_map_dir / <stem>_density.npy.
    """
    rel = item["image_path"]
    stem = Path(rel).stem
    if density_map_dir == DENSITY_MAP_DIR_AUTO:
        full_image = (root or Path(".")) / rel
        # <image_path>/../density_maps/<stem>_density.npy
        out_dir = full_image.parent.parent / "density_maps"
        return out_dir / f"{stem}_density.npy"
    return Path(density_map_dir) / f"{stem}_density.npy"


class ObjectCountingDataset(Dataset):
    """
    Dataset that yields tensors:
        image  : (3,H,W)
        density: (1,H,W)
        mask   : (1,H,W)

    Each sample is a dict with:
        - "image_path"
        - "annotation_type": "dot" | "bbox" | "segmentation"
        - "annotations"
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        root: Optional[Union[str, Path]] = None,
        *,
        density_sigma: float = 4.0,
        density_sigma_scale_bbox: float = 0.25,
        density_sigma_from_seg_area: bool = True,
        density_fixed_sigma_seg: float = 4.0,
        mask_dot_box_size: Optional[Union[int, Tuple[int, int]]] = None,
        mask_dot_sigma_to_box: float = 2.0,
        mask_dot_sigma: float = 4.0,
        mask_object_ratio: Optional[float] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        keep_original_image: bool = False,
        density_map_dir: Optional[Union[str, Path]] = DENSITY_MAP_DIR_AUTO,
        density_scale: float = DENSITY_SCALE,
    ):
        self.samples = samples
        self.density_scale = density_scale
        self.root = Path(root) if root else None
        self.density_sigma = density_sigma
        self.density_sigma_scale_bbox = density_sigma_scale_bbox
        self.density_sigma_from_seg_area = density_sigma_from_seg_area
        self.density_fixed_sigma_seg = density_fixed_sigma_seg
        self.mask_dot_box_size = mask_dot_box_size
        self.mask_dot_sigma_to_box = mask_dot_sigma_to_box
        self.mask_dot_sigma = mask_dot_sigma
        self.mask_object_ratio = mask_object_ratio
        self.transform = transform
        self.keep_original_image = keep_original_image
        if density_map_dir is None:
            self.density_map_dir = None
        elif density_map_dir == DENSITY_MAP_DIR_AUTO:
            self.density_map_dir = DENSITY_MAP_DIR_AUTO
        else:
            self.density_map_dir = Path(density_map_dir)
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        item = self.samples[idx]

        image_path = item["image_path"]
        if self.root is not None:
            image_path = self.root / image_path
        else:
            image_path = Path(image_path)

        image = _load_image(image_path)
        _, H, W = image.shape
        shape = (H, W)

        ann_type = _parse_annotation_type(item["annotation_type"])
        annotations = item["annotations"]

        if self.density_map_dir is not None:
            density_path = _density_map_path_for_sample(
                item, self.root, self.density_map_dir
            )
            if density_path.exists():
                density = np.load(str(density_path)).astype(np.float32)
                density = torch.from_numpy(density).unsqueeze(0)
            else:
                density = generate_density(
                    shape,
                    ann_type,
                    annotations,
                    sigma=self.density_sigma,
                    sigma_scale_bbox=self.density_sigma_scale_bbox,
                    sigma_from_seg_area=self.density_sigma_from_seg_area,
                    fixed_sigma_seg=self.density_fixed_sigma_seg,
                )
                density = torch.from_numpy(density.astype(np.float32)).unsqueeze(0)
                density_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(density_path), density.squeeze(0).numpy())
        else:
            density = generate_density(
                shape,
                ann_type,
                annotations,
                sigma=self.density_sigma,
                sigma_scale_bbox=self.density_sigma_scale_bbox,
                sigma_from_seg_area=self.density_sigma_from_seg_area,
                fixed_sigma_seg=self.density_fixed_sigma_seg,
            )
            density = torch.from_numpy(density.astype(np.float32)).unsqueeze(0)

        # Scale density to [0, density_scale] so the network gets larger targets (stronger gradients).
        # Count = sum(density) / density_scale.
        density = density * self.density_scale
        count = density.sum().item() / self.density_scale

        mask = generate_instance_mask(
            shape,
            ann_type,
            annotations,
            dot_box_size=self.mask_dot_box_size,
            dot_sigma_to_box=self.mask_dot_sigma_to_box,
            dot_sigma=self.mask_dot_sigma,
            mask_object_ratio=self.mask_object_ratio,
        )

        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        out = {
            "image": image,
            "density": density,
            "mask": mask,
            "annotation_type": ann_type,
            "count": torch.tensor(count, dtype=torch.float32),
        }

        if self.transform is not None:
            out = self.transform(out)

        # Keep an unmasked copy for visualization/debugging.
        if self.keep_original_image:
            out["original_image"] = out["image"].clone()

        # mask: 1 = hide (object), 0 = show. (1,H,W) * (3,H,W) -> channel-wise broadcast
        out["image"] = out["image"] * (1.0 - out["mask"].clamp(0.0, 1.0))

        return out


def precompute_density_maps(
    dataset: ObjectCountingDataset,
    density_map_dir: Optional[Union[str, Path]] = None,
) -> Union[Path, str]:
    """
    Pre-compute and save density maps for all samples so that __getitem__ can load them.

    When density_map_dir is not passed, uses the dataset's density_map_dir (e.g. DENSITY_MAP_DIR_AUTO
    for per-image default: <image_path>/../density_maps/<stem>_density.npy).
    Returns the effective directory or DENSITY_MAP_DIR_AUTO when using per-image paths.
    """
    effective = (
        Path(density_map_dir) if density_map_dir is not None and density_map_dir != DENSITY_MAP_DIR_AUTO
        else (dataset.density_map_dir if dataset.density_map_dir is not None else DENSITY_MAP_DIR_AUTO)
    )
    root = dataset.root
    if effective != DENSITY_MAP_DIR_AUTO:
        effective.mkdir(parents=True, exist_ok=True)

    for item in dataset.samples:
        path = _density_map_path_for_sample(item, root, effective)
        if path.exists():
            continue
        image_path = item["image_path"]
        if root is not None:
            full_image_path = root / image_path
        else:
            full_image_path = Path(image_path)
        image = _load_image(full_image_path)
        _, H, W = image.shape
        shape = (H, W)
        ann_type = _parse_annotation_type(item["annotation_type"])
        annotations = item["annotations"]
        density = generate_density(
            shape,
            ann_type,
            annotations,
            sigma=dataset.density_sigma,
            sigma_scale_bbox=dataset.density_sigma_scale_bbox,
            sigma_from_seg_area=dataset.density_sigma_from_seg_area,
            fixed_sigma_seg=dataset.density_fixed_sigma_seg,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), density.astype(np.float32))

    return effective