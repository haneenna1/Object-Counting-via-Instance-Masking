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
        return_instance_ids: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.samples = samples
        self.root = Path(root) if root else None
        self.density_sigma = density_sigma
        self.density_sigma_scale_bbox = density_sigma_scale_bbox
        self.density_sigma_from_seg_area = density_sigma_from_seg_area
        self.density_fixed_sigma_seg = density_fixed_sigma_seg
        self.mask_dot_box_size = mask_dot_box_size
        self.mask_dot_sigma_to_box = mask_dot_sigma_to_box
        self.mask_dot_sigma = mask_dot_sigma
        self.mask_object_ratio = mask_object_ratio
        self.return_instance_ids = return_instance_ids
        self.transform = transform

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

        # Convert outputs to tensors
        density = torch.from_numpy(density.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        out = {
            "image": image,
            "density": density,
            "mask": mask,
            "annotation_type": ann_type,
            "count": torch.tensor(len(annotations), dtype=torch.float32),
        }

        if id_map is not None:
            out["instance_ids"] = torch.from_numpy(id_map)

        if self.transform is not None:
            out = self.transform(out)

        return out