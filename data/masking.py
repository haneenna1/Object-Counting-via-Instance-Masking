"""
Instance mask generation for object counting (masked density / MAE-style training).
Produces a binary mask indicating which pixels belong to annotated instances.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

from .annotation_types import AnnotationType
from .density import compute_dot_sigmas


def _box_hw_from_scale(scale: float, aspect_hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    Integer box height and width from a scalar ``scale`` and aspect ``(h, w)``:
    ``box_h : box_w == aspect_h : aspect_w`` (same proportions as the integers).
    Scale sets the larger dimension after rounding (ties follow ``max(ah, aw)``).
    """
    ah, aw = int(aspect_hw[0]), int(aspect_hw[1])
    if ah < 1 or aw < 1:
        raise ValueError(f"dot_box_aspect must use positive integers, got {(ah, aw)}")
    m = max(ah, aw)
    box_w = max(1, int(round(scale * aw / m)))
    box_h = max(1, int(round(scale * ah / m)))
    return box_h, box_w


# ---------------------------------------------------------
# DOT HANDLERS
# ---------------------------------------------------------

def _mask_from_dots(shape, points, params) -> np.ndarray:
    """
    Binary mask: for each dot, mask a box centered at the dot.
    If ``dot_sigmas`` is provided (same length as ``points``), each box side is
    ``sigma_to_box * sigma_i`` so it matches geometry-adaptive density kernels.
    """
    H, W = shape

    box_size = params["dot_box_size"]
    sigma_to_box = params["dot_sigma_to_box"]
    sigma = params["dot_sigma"]
    per_sigmas = params.get("dot_sigmas")
    aspect_hw: Tuple[int, int] = params.get("dot_box_aspect", (1, 1))

    mask = np.zeros((H, W), dtype=np.uint8)

    for i, (x, y) in enumerate(points):
        if box_size is None:
            if per_sigmas is not None:
                s = float(per_sigmas[i])
            else:
                s = float(sigma)
            scale = float(sigma_to_box * s)
            box_h, box_w = _box_hw_from_scale(scale, aspect_hw)
        elif isinstance(box_size, int):
            box_h, box_w = _box_hw_from_scale(float(box_size), aspect_hw)
        else:
            box_h, box_w = box_size

        # print(f"box_h: {box_h}, box_w: {box_w}")
        half_h, half_w = box_h // 2, box_w // 2
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
    dot_box_aspect: Tuple[int, int] = (1, 1),
    dot_sigma_to_box: float = 2.0,
    dot_sigma: float = 4.0,
    dot_geometry_adaptive: bool = False,
    dot_geometry_beta: float = 0.3,
    dot_geometry_k: int = 3,
    dot_geometry_min_sigma: float = 4.0,
    dot_geometry_max_sigma: float = 15.0,
    mask_object_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a binary instance mask from annotations.
    If mask_object_ratio is None or 0, returns an all-zeros mask (no masking).

    dot_box_aspect:
        (height, width) integer ratio for each dot rectangle, e.g. ``(2, 1)`` → height is twice the width.
        Ignored when ``dot_box_size`` is a ``(height, width)`` tuple (explicit pixels).
    """
    H, W = shape
    if mask_object_ratio is None or mask_object_ratio == 0:
        return np.zeros((H, W), dtype=np.uint8)

    anns = list(annotations)

    dot_sigmas = None
    if annotation_type == AnnotationType.DOT and dot_box_size is None and len(anns) > 0:
        # kNN on the full annotation set so sigmas match generate_density(...)
        dot_sigmas = compute_dot_sigmas(
            anns,
            sigma=dot_sigma,
            geometry_adaptive=dot_geometry_adaptive,
            k=dot_geometry_k,
            beta=dot_geometry_beta,
            min_sigma=dot_geometry_min_sigma,
            max_sigma=dot_geometry_max_sigma,
        )
       

    # optional subsampling of objects
    if len(anns) > 0:

        r = max(0.0, min(1.0, float(mask_object_ratio)))
        k = int(round(r * len(anns)))
        k = max(0, min(len(anns), k))

        if k < len(anns):
            idx = np.random.choice(len(anns), size=k, replace=False)
            anns = [anns[i] for i in idx]
            if dot_sigmas is not None:
                dot_sigmas = dot_sigmas[idx]

    params = dict(
        dot_box_size=dot_box_size,
        dot_box_aspect=dot_box_aspect,
        dot_sigma_to_box=dot_sigma_to_box,
        dot_sigma=dot_sigma,
        dot_sigmas=dot_sigmas,
    )

    mask = MASK_GENERATORS[annotation_type](shape, anns, params)
    return mask
