"""
Instance mask generation for object counting (masked density / MAE-style training).
Produces a binary mask indicating which pixels belong to annotated instances.
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Literal

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
    Binary mask: for each dot, mask a box anchored at the dot and extending downward.
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

        half_w = box_w // 2
        ix, iy = int(round(x)), int(round(y))

        y1 = max(0, iy-half_w)
        y2 = min(H, iy -half_w + box_h)

        x1 = max(0, ix - half_w)
        x2 = min(W, ix - half_w + box_w)

        # Clip y2 so the box stops before any other dot below this one
        for j, (xj, yj) in enumerate(points):
            if j == i:
                continue
            jx, jy = int(round(xj)), int(round(yj))
            if  y1<=jy<y2 and x1<=jx<x2:
                y2 = jy
                # print(f"Clipped y2 from {y2} to {jy}")

        mask[y1:y2, x1:x2] = 1

    return mask


def _mask_from_dots_gaussian_footprint(
    shape: Tuple[int, int],
    all_points: List[Tuple[float, float]],
    masked_indices: np.ndarray,
    sigmas: np.ndarray,
    *,
    truncate_sigma: float = 2.0,
    clip_disks: bool = True,
) -> np.ndarray:
    """
    Binary mask: for each masked dot (index into ``all_points``), fill a disk of radius
    ``ceil(truncate_sigma * sigma_i)`` where ``sigma_i`` comes from ``sigmas`` (same length
    as ``all_points``), typically from :func:`compute_dot_sigmas` on the full point set.

    When ``clip_disks`` is True (default), each radius is reduced so that (i) disks for two
    masked dots do not overlap (symmetric cap at half the center distance), and (ii) no
    unmasked annotation center lies inside a masked dot's disk—so one blob does not swallow
    a neighbor head that stays visible.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if masked_indices.size == 0:
        return mask
    pts = np.asarray(all_points, dtype=np.float64)
    n = len(all_points)
    if sigmas.shape[0] != n:
        raise ValueError(
            f"sigmas length {sigmas.shape[0]} != len(all_points) {len(all_points)}"
        )
    masked_set = frozenset(int(j) for j in masked_indices.ravel())
    for i in masked_indices:
        i = int(i)
        s = float(sigmas[i])
        r = max(1, int(np.ceil(truncate_sigma * s)))
        if clip_disks:
            for j in range(n):
                if j == i:
                    continue
                dx = float(pts[i, 0] - pts[j, 0])
                dy = float(pts[i, 1] - pts[j, 1])
                d = float(np.hypot(dx, dy))
                if d <= 0.0:
                    continue
                if j in masked_set:
                    r = min(r, max(0, int(np.floor(0.5 * d - 1e-6))))
                else:
                    r = min(r, max(0, int(np.floor(d - 1e-6))))
        if r < 1:
            continue
        ix = int(round(float(pts[i, 0])))
        iy = int(round(float(pts[i, 1])))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - iy) ** 2 + (xx - ix) ** 2 <= r * r
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], disk.astype(np.uint8))

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
    dot_mask_style: Literal["box", "gaussian"] = "box",
    return_masked_indices: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate a binary instance mask from annotations.
    If mask_object_ratio is None or 0, returns an all-zeros mask (no masking).

    Always returns ``(mask, masked_idx)``. When ``return_masked_indices`` is False,
    ``masked_idx`` is None. When True, ``masked_idx`` is an int64 array of indices into
    ``annotations`` for the masked subset (empty when no masking).

    dot_box_aspect:
        (height, width) integer ratio for each dot rectangle, e.g. ``(2, 1)`` → height is twice the width.
        Ignored when ``dot_box_size`` is a ``(height, width)`` tuple (explicit pixels).

    dot_mask_style (dot annotations only):
        ``"box"``: rectangle per dot (``dot_box_size`` / sigma-scaled box), current behavior.
        ``"gaussian"``: disk mask using :func:`compute_dot_sigmas` (same hyperparameters as
        box sizing); with robust density handling, GT density subtracts the masked dots'
        Gaussians in ``dataset`` (see ``sum_dot_gaussians_for_indices``).
    """
    H, W = shape
    if dot_mask_style not in ("box", "gaussian"):
        raise ValueError(f"dot_mask_style must be 'box' or 'gaussian', got {dot_mask_style!r}")

    empty_idx = np.array([], dtype=np.int64)

    if mask_object_ratio is None or mask_object_ratio == 0:
        z = np.zeros((H, W), dtype=np.uint8)
        return z, empty_idx if return_masked_indices else None

    all_anns = list(annotations)
    n = len(all_anns)

    dot_sigmas = None
    if annotation_type == AnnotationType.DOT and dot_box_size is None and n > 0:
        # kNN on the full annotation set so sigmas match box sizing / geometry
        dot_sigmas = compute_dot_sigmas(
            all_anns,
            sigma=dot_sigma,
            geometry_adaptive=dot_geometry_adaptive,
            k=dot_geometry_k,
            beta=dot_geometry_beta,
            min_sigma=dot_geometry_min_sigma,
            max_sigma=dot_geometry_max_sigma,
        )

    r = max(0.0, min(1.0, float(mask_object_ratio)))
    k = int(round(r * n)) if n > 0 else 0
    k = max(0, min(n, k))

    if k < n:
        idx = np.random.choice(n, size=k, replace=False)
    else:
        idx = np.arange(n, dtype=np.int64)

    masked_anns = [all_anns[i] for i in idx]
    dot_sigmas_masked = dot_sigmas[idx] if dot_sigmas is not None else None

    if annotation_type == AnnotationType.DOT and dot_mask_style == "gaussian":
        g_sigmas = dot_sigmas
        if g_sigmas is None:
            g_sigmas = compute_dot_sigmas(
                all_anns,
                sigma=dot_sigma,
                geometry_adaptive=dot_geometry_adaptive,
                k=dot_geometry_k,
                beta=dot_geometry_beta,
                min_sigma=dot_geometry_min_sigma,
                max_sigma=dot_geometry_max_sigma,
            )
        mask = _mask_from_dots_gaussian_footprint(
            shape, all_anns, idx, g_sigmas
        )
    else:
        params = dict(
            dot_box_size=dot_box_size,
            dot_box_aspect=dot_box_aspect,
            dot_sigma_to_box=dot_sigma_to_box,
            dot_sigma=dot_sigma,
            dot_sigmas=dot_sigmas_masked,
        )
        mask = MASK_GENERATORS[annotation_type](shape, masked_anns, params)

    return mask, idx if return_masked_indices else None
