"""
Ground-truth density map generation from annotations.
Uses normalized 2D Gaussians so that the integral over the image approximates the object count.
"""

from functools import lru_cache

import numpy as np
from typing import List, Tuple, Union
from scipy.spatial import cKDTree as KDTree

from .annotation_types import AnnotationType


# ---------------------------------------------------------
# GAUSSIAN UTILITY
# ---------------------------------------------------------

_GAUSS_TRUNCATE = 3.0
_SIGMA_QUANT = 0.25


@lru_cache(maxsize=1024)
def _cached_gaussian_kernel(sigma_q: float, truncate: float) -> np.ndarray:
    """Normalized 2-D Gaussian kernel for quantized sigma; cached across heads."""
    rad = max(1, int(np.ceil(truncate * sigma_q)))
    y = np.arange(-rad, rad + 1, dtype=np.float64)
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    g = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma_q ** 2))
    g /= g.sum()
    return g.astype(np.float32)


def _accumulate_gaussian(
    out: np.ndarray,
    H: int,
    W: int,
    y_int: int,
    x_int: int,
    sigma: float,
    *,
    truncate: float = _GAUSS_TRUNCATE,
    quant: float = _SIGMA_QUANT,
) -> None:
    """
    Add a normalized Gaussian blob for a single head, using a cached analytic
    kernel (quantized to ``quant`` steps).
    """
    if sigma <= 0:
        return
    sigma_q = round(sigma / quant) * quant
    if sigma_q <= 0:
        sigma_q = quant
    g = _cached_gaussian_kernel(sigma_q, truncate)
    rad = g.shape[0] // 2
    y0 = max(0, y_int - rad)
    y1 = min(H, y_int + rad + 1)
    x0 = max(0, x_int - rad)
    x1 = min(W, x_int + rad + 1)
    if y0 >= y1 or x0 >= x1:
        return
    gy0 = y0 - (y_int - rad)
    gx0 = x0 - (x_int - rad)
    gy1 = gy0 + (y1 - y0)
    gx1 = gx0 + (x1 - x0)
    out[y0:y1, x0:x1] += g[gy0:gy1, gx0:gx1]


# ---------------------------------------------------------
# DOT SIGMAS (shared with instance masking)
# ---------------------------------------------------------

@lru_cache(maxsize=8192)
def _cached_knn_dists(pts_bytes: bytes, n: int, k: int, leafsize: int) -> np.ndarray:
    """k-NN distance rows cached per (point-set, k)."""
    pts = np.frombuffer(pts_bytes, dtype=np.float64).reshape(n, 2)
    tree = KDTree(pts.copy(), leafsize=leafsize)
    kq = min(k + 1, n)
    dists, _ = tree.query(pts, k=kq)
    if dists.ndim == 1:
        dists = dists.reshape(n, 1)
    return dists.astype(np.float64, copy=False)


def compute_dot_sigmas(
    points: List[Tuple[float, float]],
    *,
    sigma: float = 4.0,
    geometry_adaptive: bool = True,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float = 15.0,
    kdtree_leafsize: int = 2048,
) -> np.ndarray:
    """
    Per-point Gaussian sigmas for dot annotations.

    geometry_adaptive (when len(points) > k): sigma_i = beta * mean_kNN_distance_i.
    Otherwise: fixed ``sigma`` for every point.
    """
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    pts = np.asarray(points, dtype=np.float64)

    if geometry_adaptive and n > k:
        dists_all = _cached_knn_dists(pts.tobytes(), n, k, int(kdtree_leafsize))
        dists = dists_all[:, 1 : k + 1]
        sigmas = beta * dists.mean(axis=1)
    else:
        sigmas = np.full(n, sigma, dtype=np.float64)

    return sigmas


# ---------------------------------------------------------
# DOT HANDLER
# ---------------------------------------------------------

def _density_from_points(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    sigma: float = 4.0,
    geometry_adaptive: bool = True,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float = 15.0,
    **kwargs,
) -> np.ndarray:
    """
    Density map from dot annotations using geometry-adaptive Gaussian kernels.
    """
    H, W = shape
    density = np.zeros((H, W), dtype=np.float32)

    if len(points) == 0:
        return density

    pts = np.asarray(points, dtype=np.float64)
    sigmas = compute_dot_sigmas(
        points,
        sigma=sigma,
        geometry_adaptive=geometry_adaptive,
        k=k,
        beta=beta,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
    )

    for i in range(len(pts)):
        x_int = int(round(float(pts[i, 0])))
        y_int = int(round(float(pts[i, 1])))
        if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H:
            continue
        _accumulate_gaussian(density, H, W, y_int, x_int, float(sigmas[i]))

    return density


def sum_dot_gaussians_for_indices(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    indices: Union[List[int], np.ndarray],
    *,
    sigma: float = 4.0,
    geometry_adaptive: bool = True,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float = 15.0,
) -> np.ndarray:
    """
    Sum of density contributions for ``points[i]`` with ``i`` in ``indices``.
    Sigmas are computed from k-NN on the **full** ``points`` list (same as
    :func:`_density_from_points`), so subtraction from the full density map is exact.
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.float32)
    n = len(points)
    if n == 0 or len(indices) == 0:
        return out

    pts = np.asarray(points, dtype=np.float64)
    sigmas = compute_dot_sigmas(
        points,
        sigma=sigma,
        geometry_adaptive=geometry_adaptive,
        k=k,
        beta=beta,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
    )

    for i in indices:
        i = int(i)
        x_int = int(round(float(pts[i, 0])))
        y_int = int(round(float(pts[i, 1])))
        if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H:
            continue
        _accumulate_gaussian(out, H, W, y_int, x_int, float(sigmas[i]))

    return out


# ---------------------------------------------------------
# BBOX HANDLER
# ---------------------------------------------------------

def _density_from_bboxes(shape, bboxes, sigma_scale_bbox=0.25, **kwargs):
    sigma_scale = sigma_scale_bbox
    H, W = shape
    density = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2) in bboxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        sigma = max(sigma_scale * (w + h) / 2.0, 1.0)
        _accumulate_gaussian(density, H, W, int(round(cy)), int(round(cx)), sigma)

    return density


# ---------------------------------------------------------
# SEGMENTATION HANDLER
# ---------------------------------------------------------

def _density_from_segmentations(shape, masks, sigma_from_seg_area=True, fixed_sigma_seg=4.0, **kwargs):
    sigma_from_area = sigma_from_seg_area
    fixed_sigma = fixed_sigma_seg
    H, W = shape
    density = np.zeros((H, W), dtype=np.float32)

    for mask in masks:
        if mask.shape != (H, W):
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {(H, W)}")
        binary = (mask > 0).astype(np.float64)
        area = binary.sum()
        if area <= 0:
            continue
        rows, cols = np.where(binary > 0)
        cy, cx = rows.mean(), cols.mean()
        if sigma_from_area:
            sigma = max(np.sqrt(area) * 0.5, 1.0)
        else:
            sigma = fixed_sigma
        _accumulate_gaussian(density, H, W, int(round(cy)), int(round(cx)), sigma)

    return density


# ---------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------

DENSITY_GENERATORS = {
    AnnotationType.DOT: _density_from_points,
    AnnotationType.BBOX: _density_from_bboxes,
    AnnotationType.SEGMENTATION: _density_from_segmentations,
}


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def generate_density(
    shape: Tuple[int, int],
    annotation_type: AnnotationType,
    annotations: Union[
        List[Tuple[float, float]],
        List[Tuple[float, float, float, float]],
        List[np.ndarray],
    ],
    *,
    sigma: float = 4.0,
    sigma_scale_bbox: float = 0.25,
    sigma_from_seg_area: bool = True,
    fixed_sigma_seg: float = 4.0,
    geometry_adaptive: bool = False,
    beta: float = 0.3,
    k: int = 3,
    min_sigma: float = 4.0,
) -> np.ndarray:
    """
    Generate ground-truth density map from annotations.

    For dot annotations, sigma controls spread and peak:
    - Smaller sigma -> narrower blob (less spread), higher peak at the dot.
    - Larger sigma -> wider blob (more spread), lower peak.
    Peak value at center = 1 / (2*pi*sigma^2); integral over plane = 1 per dot.

    geometry_adaptive:
      sigma_i = beta * d_i
      where d_i is the mean distance to the k nearest neighbors of point i.
      For sparse crowds (len(points) <= k), this falls back to fixed sigma.
    """
    params = dict(
        sigma=sigma,
        sigma_scale_bbox=sigma_scale_bbox,
        sigma_from_seg_area=sigma_from_seg_area,
        fixed_sigma_seg=fixed_sigma_seg,
        geometry_adaptive=geometry_adaptive,
        beta=beta,
        k=k,
        min_sigma=min_sigma,
    )

    try:
        handler = DENSITY_GENERATORS[annotation_type]
    except KeyError:
        raise ValueError(f"Unsupported annotation_type: {annotation_type}")

    return handler(shape, annotations, **params)