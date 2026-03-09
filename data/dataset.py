"""
Unified object counting dataset: loads images and produces density maps and instance masks
according to annotation type (dot, bbox, segmentation).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .annotation_types import AnnotationType
from .density import generate_density
from .masking import generate_instance_mask

import torch
from torch.utils.data import Dataset as TorchDataset


# Type aliases for annotations
DotAnnotations = List[Tuple[float, float]]
BboxAnnotations = List[Tuple[float, float, float, float]]
SegmentationAnnotations = List[np.ndarray]


def _load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image as (H, W, C) uint8 RGB."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    return np.asarray(im)


def _parse_annotation_type(value: Any) -> AnnotationType:
    if isinstance(value, AnnotationType):
        return value
    if isinstance(value, str):
        return AnnotationType(value.lower())
    raise ValueError(f"Cannot parse annotation type: {value}")


class ObjectCountingDataset:
    """
    Dataset that yields (image, density_map, instance_mask) with logic determined by annotation type.

    Each sample is a dict with:
        - "image_path": path to image file
        - "annotation_type": "dot" | "bbox" | "segmentation"
        - "annotations": list of (x,y), (x1,y1,x2,y2), or (H,W) masks
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
        return_instance_ids: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Args:
            samples: List of dicts with "image_path", "annotation_type", "annotations".
            root: Optional root directory to prepend to image_path.
            density_sigma: Gaussian sigma for dot annotations.
            density_sigma_scale_bbox: Sigma scale for bbox (sigma = scale * (w+h)/2).
            density_sigma_from_seg_area: For segmentation, derive sigma from area.
            density_fixed_sigma_seg: For segmentation, fixed sigma when not from area.
            mask_dot_box_size: Box size for dot masking (None = from sigma).
            mask_dot_sigma_to_box: For dot mask box when mask_dot_box_size is None.
            mask_dot_sigma: Sigma used for dot mask box size when mask_dot_box_size is None.
            mask_object_ratio: Optional[0..1]; fraction of objects to mask per image (random subset).
            return_instance_ids: If True, __getitem__ also returns instance id map.
            transform: Optional callable(sample_dict) -> sample_dict; receives and returns dict with
                       keys "image", "density", "mask", etc., for augmentation.
        """
        self.samples = samples
        self.root = Path(root) if root else None
        self.density_sigma = density_sigma
        self.density_sigma_scale_bbox = density_sigma_scale_bbox
        self.density_sigma_from_seg_area = density_sigma_from_seg_area
        self.density_fixed_sigma_seg = density_fixed_sigma_seg
        self.mask_dot_box_size = mask_dot_box_size
        self.mask_dot_sigma_to_box = mask_dot_sigma_to_box
        self.mask_dot_sigma = mask_dot_sigma
        # default masking ratio (0..1) for all images
        self.mask_object_ratio = mask_object_ratio
        self.return_instance_ids = return_instance_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _get_item_raw(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx].copy()
        image_path = item["image_path"]
        if self.root is not None:
            image_path = self.root / image_path
        else:
            image_path = Path(image_path)

        image = _load_image(image_path)
        H, W = image.shape[:2]
        shape = (H, W)

        ann_type = _parse_annotation_type(item["annotation_type"])
        annotations = item["annotations"]

        density = generate_density(
            shape,
            ann_type,
            annotations,
            sigma=self.density_sigma,
            sigma_scale_bbox=self.density_sigma_scale_bbox,
            sigma_from_seg_area=self.density_sigma_from_seg_area,
            fixed_sigma_seg=self.density_fixed_sigma_seg,
        )

        mask_result = generate_instance_mask(
            shape,
            ann_type,
            annotations,
            dot_box_size=self.mask_dot_box_size,
            dot_sigma_to_box=self.mask_dot_sigma_to_box,
            dot_sigma=self.mask_dot_sigma,
            mask_object_ratio=self.mask_object_ratio,
            return_instance_ids=self.return_instance_ids,
        )

        if self.return_instance_ids:
            mask, id_map = mask_result
        else:
            mask = mask_result
            id_map = None

        out = {
            "image": image,
            "density": density,
            "mask": mask,
            "annotation_type": ann_type,
            "count": len(annotations),
        }
        if id_map is not None:
            out["instance_ids"] = id_map
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = self._get_item_raw(idx)
        if self.transform is not None:
            out = self.transform(out)
        return out

class TorchObjectCountingDataset(TorchDataset):
    """
    PyTorch Dataset wrapper: returns tensors (image, density, mask) in CHW format.
    """

    def __init__(self, base: ObjectCountingDataset, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = self.base[idx]
        # HWC -> CHW, numpy -> torch
        image = torch.from_numpy(out["image"]).permute(2, 0, 1).float() / 255.0
        density = torch.from_numpy(out["density"].astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(out["mask"]).unsqueeze(0).float()
        result = {
            "image": image,
            "density": density,
            "mask": mask,
            "count": out["count"],
            "annotation_type": out["annotation_type"],
        }
        if "instance_ids" in out:
            result["instance_ids"] = torch.from_numpy(out["instance_ids"])
        return result
