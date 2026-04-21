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
    n = len(points)
    if n == 0:
        return mask

    # Precompute integer pixel coords of every dot once (was recomputed
    # inside an O(N^2) Python loop per image before).
    pts_arr = np.asarray(points, dtype=np.float64)
    ix_all = np.rint(pts_arr[:, 0]).astype(np.int64)
    iy_all = np.rint(pts_arr[:, 1]).astype(np.int64)

    # Precompute (box_h, box_w) per dot so the inner loop is minimal.
    if box_size is None:
        if per_sigmas is not None:
            per_s = np.asarray(per_sigmas, dtype=np.float64)
        else:
            per_s = np.full(n, float(sigma), dtype=np.float64)
        scales = sigma_to_box * per_s
        box_hw = [_box_hw_from_scale(float(s), aspect_hw) for s in scales]
    elif isinstance(box_size, int):
        bh, bw = _box_hw_from_scale(float(box_size), aspect_hw)
        box_hw = [(bh, bw)] * n
    else:
        bh, bw = int(box_size[0]), int(box_size[1])
        box_hw = [(bh, bw)] * n

    for i in range(n):
        box_h, box_w = box_hw[i]
        half_w = box_w // 2
        ix = int(ix_all[i])
        iy = int(iy_all[i])

        y1 = max(0, iy - half_w)
        y2 = min(H, iy - half_w + box_h)
        x1 = max(0, ix - half_w)
        x2 = min(W, ix - half_w + box_w)

        # Clip y2 so the box stops before any other dot below this one.
        # Vectorized across all points (excluding self) instead of a Python loop.
        # The original loop monotonically reduces y2 down to the minimum jy
        # satisfying the bounds, so we compute that minimum directly.
        if y2 > y1 and x2 > x1:
            in_range = (
                (iy_all >= y1) & (iy_all < y2) & (ix_all >= x1) & (ix_all < x2)
            )
            in_range[i] = False
            if in_range.any():
                y2 = int(iy_all[in_range].min())

            if y2 > y1:
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

    masked_idx = np.asarray(masked_indices, dtype=np.int64).ravel()
    m = masked_idx.size

    # Initial per-disk radii before clipping: max(1, ceil(truncate_sigma * sigma_i))
    sigmas_m = np.asarray(sigmas, dtype=np.float64)[masked_idx]
    radii = np.maximum(1, np.ceil(truncate_sigma * sigmas_m).astype(np.int64))

    if clip_disks and n > 1:
        # Vectorize the O(m * n) Python loop: compute pairwise distances
        # between masked points and all points, then reduce. This replaces
        # ~m*n scalar hypot / floor calls per image with a single numpy op.
        pm = pts[masked_idx]  # (m, 2)
        dx = pm[:, 0:1] - pts[None, :, 0]  # (m, n)
        dy = pm[:, 1:2] - pts[None, :, 1]  # (m, n)
        d = np.hypot(dx, dy)  # (m, n)

        # Exclude self-comparisons and coincident points (original: `if d <= 0: continue`).
        # Any pair with d == 0 is treated as "skip", including i == j.
        valid = d > 0.0
        d_valid = np.where(valid, d, np.inf)

        # Split neighbors into masked / unmasked columns.
        is_masked_col = np.zeros(n, dtype=bool)
        is_masked_col[masked_idx] = True

        eps = 1e-6
        d_to_masked = np.where(is_masked_col[None, :], d_valid, np.inf)
        d_to_unmasked = np.where(~is_masked_col[None, :], d_valid, np.inf)

        min_to_masked = d_to_masked.min(axis=1)  # (m,)
        min_to_unmasked = d_to_unmasked.min(axis=1)  # (m,)

        # Matches original: r = min(r, max(0, floor(0.5 * d - eps))) for masked j,
        #                   r = min(r, max(0, floor(d - eps)))       for unmasked j.
        cap_m = np.where(
            np.isfinite(min_to_masked),
            np.maximum(0.0, np.floor(0.5 * min_to_masked - eps)),
            np.inf,
        )
        cap_u = np.where(
            np.isfinite(min_to_unmasked),
            np.maximum(0.0, np.floor(min_to_unmasked - eps)),
            np.inf,
        )
        combined_cap = np.minimum(cap_m, cap_u)
        # combined_cap may be inf (no neighbors); np.minimum(finite_int, inf) stays finite.
        radii = np.minimum(radii.astype(np.float64), combined_cap).astype(np.int64)

    # Rasterize disks (kept as a Python loop; hot path was the O(N^2) clipping above).
    ix_all = np.rint(pts[:, 0]).astype(np.int64)
    iy_all = np.rint(pts[:, 1]).astype(np.int64)
    for k in range(m):
        r = int(radii[k])
        if r < 1:
            continue
        i = int(masked_idx[k])
        ix = int(ix_all[i])
        iy = int(iy_all[i])
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - iy) ** 2 + (xx - ix) ** 2 <= r * r
        np.bitwise_or(mask[y0:y1, x0:x1], disk.view(np.uint8), out=mask[y0:y1, x0:x1])

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
    rng: Optional[np.random.Generator] = None,
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
        if rng is None:
            idx = np.random.choice(n, size=k, replace=False)
        else:
            idx = rng.choice(n, size=k, replace=False)
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
