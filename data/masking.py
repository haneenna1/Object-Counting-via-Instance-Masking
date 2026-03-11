"""
Instance mask generation for object counting (masked density / MAE-style training).
Produces a binary mask indicating which pixels belong to annotated instances.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

from .annotation_types import AnnotationType


# ---------------------------------------------------------
# DOT HANDLERS
# ---------------------------------------------------------

def _mask_from_dots(shape, points, params) -> np.ndarray:
    """
    Binary mask: for each dot, mask a box centered at the dot.
    """
    H, W = shape

    box_size = params["dot_box_size"]
    sigma_to_box = params["dot_sigma_to_box"]
    sigma = params["dot_sigma"]

    if box_size is None:
        side = max(1, int(round(sigma_to_box * sigma)))
        box_h = box_w = side
    elif isinstance(box_size, int):
        box_h = box_w = box_size
    else:
        box_h, box_w = box_size

    mask = np.zeros((H, W), dtype=np.uint8)

    half_h, half_w = box_h // 2, box_w // 2

    for (x, y) in points:
        ix, iy = int(round(x)), int(round(y))

        y1 = max(0, iy - half_h)
        y2 = min(H, iy - half_h + box_h)

        x1 = max(0, ix - half_w)
        x2 = min(W, ix - half_w + box_w)

        mask[y1:y2, x1:x2] = 1

    return mask


# ---------------------------------------------------------
# BBOX HANDLERS
# ---------------------------------------------------------

def _mask_from_bboxes(shape, bboxes, params) -> np.ndarray:
    """Binary mask: union of all bounding box regions."""
    H, W = shape

    mask = np.zeros((H, W), dtype=np.uint8)

    for (x1, y1, x2, y2) in bboxes:

        x1, x2 = int(max(0, x1)), int(min(W, x2))
        y1, y2 = int(max(0, y1)), int(min(H, y2))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return mask


# ---------------------------------------------------------
# SEGMENTATION HANDLERS
# ---------------------------------------------------------

def _mask_from_segmentations(shape, masks, params) -> np.ndarray:
    """Binary mask: union of all instance segmentation masks."""
    H, W = shape

    mask = np.zeros((H, W), dtype=np.uint8)

    for m in masks:

        if m.shape != (H, W):
            raise ValueError(f"Mask shape {m.shape} != {(H, W)}")

        mask = np.maximum(mask, (m > 0).astype(np.uint8))

    return mask


# ---------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------

MASK_GENERATORS = {
    AnnotationType.DOT: _mask_from_dots,
    AnnotationType.BBOX: _mask_from_bboxes,
    AnnotationType.SEGMENTATION: _mask_from_segmentations,
}


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def generate_instance_mask(
    shape: Tuple[int, int],
    annotation_type: AnnotationType,
    annotations: Union[
        List[Tuple[float, float]],
        List[Tuple[float, float, float, float]],
        List[np.ndarray],
    ],
    *,
    dot_box_size: Optional[Union[int, Tuple[int, int]]] = None,
    dot_sigma_to_box: float = 2.0,
    dot_sigma: float = 4.0,
    mask_object_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a binary instance mask from annotations.
    """
    anns = list(annotations)

    # optional subsampling of objects
    if mask_object_ratio is not None and len(anns) > 0:

        r = max(0.0, min(1.0, float(mask_object_ratio)))
        k = int(round(r * len(anns)))
        k = max(0, min(len(anns), k))

        if k < len(anns):
            idx = np.random.choice(len(anns), size=k, replace=False)
            anns = [anns[i] for i in idx]

    params = dict(
        dot_box_size=dot_box_size,
        dot_sigma_to_box=dot_sigma_to_box,
        dot_sigma=dot_sigma,
    )

    mask = MASK_GENERATORS[annotation_type](shape, anns, params)
    return mask
