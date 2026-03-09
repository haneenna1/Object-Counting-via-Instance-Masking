"""
Instance mask generation for object counting (masked density / MAE-style training).
Produces a binary mask (or instance-id map) indicating which pixels belong to annotated instances.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

from .annotation_types import AnnotationType


def _mask_from_dots(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    box_size: Optional[Union[int, Tuple[int, int]]] = None,
    sigma_to_box: float = 2.0,
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Binary mask: for each dot, mask a box centered at the dot.
    box_size: (h, w) or single int for square. If None, derived from sigma (e.g. 2*sigma).
    """
    H, W = shape
    if box_size is None:
        side = max(1, int(round(sigma_to_box * sigma)))
        box_h = box_w = side
    elif isinstance(box_size, int):
        box_h = box_w = box_size
    else:
        box_h, box_w = box_size
    print("masking from dots - amount is: ", len(points))
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


def _mask_from_bboxes(
    shape: Tuple[int, int],
    bboxes: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Binary mask: union of all bounding box regions. Boxes in (x1, y1, x2, y2)."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        x1, x2 = int(max(0, x1)), int(min(W, x2))
        y1, y2 = int(max(0, y1)), int(min(H, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def _mask_from_segmentations(
    shape: Tuple[int, int],
    masks: List[np.ndarray],
) -> np.ndarray:
    """Binary mask: union of all instance segmentation masks."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        if m.shape != (H, W):
            raise ValueError(f"Mask shape {m.shape} != {(H, W)}")
        mask = np.maximum(mask, (m > 0).astype(np.uint8))
    return mask


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
    return_instance_ids: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Generate instance mask (binary or instance-id map) from annotations.

    Args:
        shape: (H, W) of the image.
        annotation_type: One of AnnotationType.DOT, BBOX, SEGMENTATION.
        annotations: Same as in generate_density (dots, bboxes, or masks).
        dot_box_size: For DOT: (h, w) or int. If None, box size = 2 * dot_sigma.
        dot_sigma_to_box: For DOT: box side = dot_sigma_to_box * dot_sigma when dot_box_size is None.
        dot_sigma: For DOT: sigma used to derive box size when dot_box_size is None.
        return_instance_ids: If True, also return (H, W) with instance id per pixel (0 = background).

    Returns:
        mask: (H, W) uint8, 1 where instances are masked.
        If return_instance_ids: (mask, id_map) where id_map has 1..N per instance.
    """
    H, W = shape
    id_map = np.zeros((H, W), dtype=np.int32) if return_instance_ids else None

    # Optionally subsample objects to control masking ratio
    anns = list(annotations)
    if mask_object_ratio is not None and len(anns) > 0:
        r = max(0.0, min(1.0, float(mask_object_ratio)))
        k = int(round(r * len(anns)))
        k = max(0, min(len(anns), k))
        if k < len(anns):
            idx = np.random.choice(len(anns), size=k, replace=False)
            anns = [anns[i] for i in idx]

    if annotation_type == AnnotationType.DOT:
        mask = _mask_from_dots(
            shape,
            anns,
            box_size=dot_box_size,
            sigma_to_box=dot_sigma_to_box,
            sigma=dot_sigma,
        )
        if return_instance_ids:
            if dot_box_size is None:
                side = max(1, int(round(dot_sigma_to_box * dot_sigma)))
                box_h = box_w = side
            elif isinstance(dot_box_size, int):
                box_h = box_w = dot_box_size
            else:
                box_h, box_w = dot_box_size[0], dot_box_size[1]
            half_h, half_w = box_h // 2, box_w // 2
            for i, (x, y) in enumerate(anns):
                ix, iy = int(round(x)), int(round(y))
                y1 = max(0, iy - half_h)
                y2 = min(H, iy - half_h + box_h)
                x1 = max(0, ix - half_w)
                x2 = min(W, ix - half_w + box_w)
                id_map[y1:y2, x1:x2] = i + 1
        return (mask, id_map) if return_instance_ids else mask

    if annotation_type == AnnotationType.BBOX:
        mask = _mask_from_bboxes(shape, anns)
        if return_instance_ids:
            for i, (x1, y1, x2, y2) in enumerate(anns):
                x1, x2 = int(max(0, x1)), int(min(W, x2))
                y1, y2 = int(max(0, y1)), int(min(H, y2))
                if x2 > x1 and y2 > y1:
                    id_map[y1:y2, x1:x2] = i + 1
        return (mask, id_map) if return_instance_ids else mask

    if annotation_type == AnnotationType.SEGMENTATION:
        mask = _mask_from_segmentations(shape, anns)
        if return_instance_ids:
            for i, m in enumerate(anns):
                id_map[(m > 0)] = i + 1
        return (mask, id_map) if return_instance_ids else mask

    raise ValueError(f"Unsupported annotation_type: {annotation_type}")
