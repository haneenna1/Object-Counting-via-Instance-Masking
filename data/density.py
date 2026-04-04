"""
Ground-truth density map generation from annotations.
Uses normalized 2D Gaussians so that the integral over the image approximates the object count.
"""

from functools import lru_cache

import numpy as np
from typing import List, Tuple, Union
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

from .annotation_types import AnnotationType


# ---------------------------------------------------------
# GAUSSIAN UTILITY
# ---------------------------------------------------------

def gaussian_2d_patch(
    sigma: float,
    truncate: float = 3.0,
) -> np.ndarray:
    """
    Create a normalized 2D Gaussian patch centered at (0,0),
    truncated at `truncate * sigma`.
    """
    radius = int(truncate * sigma)
    size = 2 * radius + 1

    y = np.arange(-radius, radius + 1, dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    g = np.exp(-(yy**2 + xx**2) / (2 * sigma**2))

    # ✅ Discrete normalization (critical)
    g /= g.sum()

    return g


@lru_cache(maxsize=8192)
def _cached_csrnet_neighbor_dists(pts_bytes: bytes, n: int, leafsize: int) -> np.ndarray:
    """KDTree k-NN rows for CSRNet-style sigma (and default dot geometry when k<=3)."""
    pts = np.frombuffer(pts_bytes, dtype=np.float64).reshape(n, 2)
    tree = KDTree(pts.copy(), leafsize=leafsize)
    kq = min(4, n)
    dists, _ = tree.query(pts, k=kq)
    if dists.ndim == 1:
        dists = dists.reshape(n, 1)
    return dists.astype(np.float64, copy=False)


def csrnet_neighbor_dists(
    points: List[Tuple[float, float]],
    *,
    kdtree_leafsize: int = 2048,
) -> np.ndarray:
    """
    Distance matrix: column 0 is self (0), columns 1.. are nearest neighbors
    (up to three when n>=4). Cached per point-set to avoid repeated KDTree work
    in ``__getitem__`` (mask + CSRNet removal + density build).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return _cached_csrnet_neighbor_dists(pts.tobytes(), n, int(kdtree_leafsize))


# ---------------------------------------------------------
# DOT SIGMAS (shared with instance masking)
# ---------------------------------------------------------

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
    Per-point Gaussian sigmas for dot annotations (same rule as density map generation).

    geometry_adaptive (when len(points) > k): sigma_i = beta * mean_kNN_distance_i.
    Otherwise: fixed ``sigma`` for every point.
    """
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    pts = np.asarray(points, dtype=np.float64)

    if geometry_adaptive and n > k:
        kq = min(4, n)
        if k + 1 <= kq:
            dists_all = csrnet_neighbor_dists(points, kdtree_leafsize=kdtree_leafsize)
            dists = dists_all[:, 1 : k + 1]
            mean_dists = dists.mean(axis=1)
            sigmas = beta * mean_dists
        else:
            tree = KDTree(pts, leafsize=kdtree_leafsize)
            dists, _ = tree.query(pts, k=k + 1)
            dists = dists[:, 1:]
            mean_dists = dists.mean(axis=1)
            sigmas = beta * mean_dists
        # sigmas = np.clip(sigmas, min_sigma, max_sigma)
    else:
        sigmas = np.full(n, sigma, dtype=np.float64)

    return sigmas


# ---------------------------------------------------------
# DOT HANDLER
# ---------------------------------------------------------

def _density_from_points(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],  # (x, y)
    sigma: float = 4.0,
    geometry_adaptive: bool = True,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float = 15.0,
    **kwargs,
) -> np.ndarray:
    """
    Generate density map using geometry-adaptive kernels (CSRNet/MCNN style).

    Args:
        shape: (H, W)
        points: list of (x, y)
    """
    H, W = shape
    density = np.zeros((H, W), dtype=np.float64)

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
    # -----------------------------------------------------
    # Place Gaussians (local patches)
    # -----------------------------------------------------
    for (x, y), s in zip(pts, sigmas):
        x_int = int(round(x))
        y_int = int(round(y))

        # Skip invalid points
        if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H:
            continue

        g = gaussian_2d_patch(s)
        radius = g.shape[0] // 2

        # Bounding box in image
        y1 = max(0, y_int - radius)
        y2 = min(H, y_int + radius + 1)
        x1 = max(0, x_int - radius)
        x2 = min(W, x_int + radius + 1)

        # Corresponding region in Gaussian
        gy1 = y1 - (y_int - radius)
        gy2 = gy1 + (y2 - y1)
        gx1 = x1 - (x_int - radius)
        gx2 = gx1 + (x2 - x1)

        density[y1:y2, x1:x2] += g[gy1:gy2, gx1:gx2]

    return density


def _csrnet_reference_sigma(
    dist_row: np.ndarray,
    n_points: int,
    shape: Tuple[int, int],
) -> float:
    """
    Per-head sigma from the official CSRNet-pytorch ``make_dataset.ipynb`` recipe:
    ``sigma = (d1 + d2 + d3) * 0.1`` for three nearest neighbors (excluding self).
    Single point: ``mean(H, W) / 4`` (same as ``average(shape) / 2 / 2`` in the notebook).
    For fewer than three neighbors (tiny point sets), approximate with available distances.
    """
    if n_points <= 1:
        return float(np.mean(shape) / 4.0)

    nn = dist_row[1:]
    if nn.size >= 3:
        return float(0.1 * (float(nn[0]) + float(nn[1]) + float(nn[2])))
    if nn.size == 2:
        return float(0.1 * (float(nn[0]) + float(nn[1]) + float(nn[1])))
    if nn.size == 1:
        return float(0.3 * float(nn[0]))
    return float(np.mean(shape) / 4.0)


# Match scipy.ndimage.gaussian_filter(..., mode="constant") default truncation.
_CSRNET_GAUSS_TRUNCATE = 4.0


def _accumulate_csrnet_impulse_gaussian(
    out: np.ndarray,
    H: int,
    W: int,
    y_int: int,
    x_int: int,
    sigma: float,
    *,
    truncate: float = _CSRNET_GAUSS_TRUNCATE,
) -> None:
    """
    Add ``gaussian_filter(impulse, sigma, mode='constant')`` for a single pixel impulse,
    using a local patch so each head costs O((truncate*σ)²) instead of O(H*W).
    """
    if sigma <= 0:
        return
    rad = max(1, int(np.ceil(truncate * float(sigma))))
    y0 = max(0, y_int - rad)
    y1 = min(H, y_int + rad + 1)
    x0 = max(0, x_int - rad)
    x1 = min(W, x_int + rad + 1)
    ph, pw = y1 - y0, x1 - x0
    if ph <= 0 or pw <= 0:
        return
    ly, lx = y_int - y0, x_int - x0
    impulse = np.zeros((ph, pw), dtype=np.float32)
    impulse[ly, lx] = 1.0
    blob = gaussian_filter(
        impulse, sigma, mode="constant", truncate=truncate
    )
    out[y0:y1, x0:x1] += blob.astype(np.float32, copy=False)


def density_from_points_csrnet_reference(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    *,
    kdtree_leafsize: int = 2048,
    **kwargs,
) -> np.ndarray:
    """
    CSRNet reference preprocessing (``make_dataset.ipynb`` / CrowdNet-style):
    for each head, place a unit impulse on the rounded pixel and accumulate
    ``scipy.ndimage.gaussian_filter(impulse, sigma, mode='constant')``,
    with adaptive ``sigma`` from k-NN distances (0.1 * sum of three NN distances).

    This is **not** the same as :func:`_density_from_points` (discrete normalized patches).
    """
    H, W = shape
    density = np.zeros((H, W), dtype=np.float32)
    n = len(points)
    if n == 0:
        return density

    pts = np.asarray(points, dtype=np.float64)
    dists_all = csrnet_neighbor_dists(points, kdtree_leafsize=kdtree_leafsize)

    for i in range(n):
        x_int = int(round(float(pts[i, 0])))
        y_int = int(round(float(pts[i, 1])))
        if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H:
            continue

        sigma = _csrnet_reference_sigma(dists_all[i], n, shape)
        if sigma <= 0:
            sigma = float(np.mean(shape) / 4.0)

        _accumulate_csrnet_impulse_gaussian(
            density, H, W, y_int, x_int, sigma
        )

    return density


def sum_csrnet_dot_gaussians_for_indices(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    indices: Union[List[int], np.ndarray],
    *,
    kdtree_leafsize: int = 2048,
) -> np.ndarray:
    """
    Sum of single-head CSRNet-style density contributions for ``points[i]`` with ``i`` in ``indices``.
    Sigmas are computed from k-NN on the full ``points`` list (same as the full GT map).
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.float32)
    n = len(points)
    if n == 0 or len(indices) == 0:
        return out
    pts = np.asarray(points, dtype=np.float64)
    dists_all = csrnet_neighbor_dists(points, kdtree_leafsize=kdtree_leafsize)
    for i in indices:
        i = int(i)
        x_int = int(round(float(pts[i, 0])))
        y_int = int(round(float(pts[i, 1])))
        if x_int < 0 or x_int >= W or y_int < 0 or y_int >= H:
            continue
        sigma = _csrnet_reference_sigma(dists_all[i], n, shape)
        if sigma <= 0:
            sigma = float(np.mean(shape) / 4.0)
        _accumulate_csrnet_impulse_gaussian(out, H, W, y_int, x_int, sigma)
    return out


# ---------------------------------------------------------
# BBOX HANDLER
# ---------------------------------------------------------

def _density_from_bboxes(shape, bboxes, sigma_scale_bbox=0.25, **kwargs):

    sigma_scale = sigma_scale_bbox

    H, W = shape
    density = np.zeros((H, W), dtype=np.float64)

    for (x1, y1, x2, y2) in bboxes:

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        sigma = sigma_scale * (w + h) / 2.0
        sigma = max(sigma, 1.0)

        density += gaussian_2d_patch(sigma)

    return density


# ---------------------------------------------------------
# SEGMENTATION HANDLER
# ---------------------------------------------------------

def _density_from_segmentations(shape, masks, sigma_from_seg_area=True, fixed_sigma_seg=4.0, **kwargs):

    sigma_from_area = sigma_from_seg_area
    fixed_sigma = fixed_sigma_seg

    H, W = shape
    density = np.zeros((H, W), dtype=np.float64)

    for mask in masks:

        if mask.shape != (H, W):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {(H, W)}"
            )

        binary = (mask > 0).astype(np.float64)
        area = binary.sum()

        if area <= 0:
            continue

        rows, cols = np.where(binary > 0)

        cy = rows.mean()
        cx = cols.mean()

        if sigma_from_area:
            sigma = np.sqrt(area) * 0.5
            sigma = max(sigma, 1.0)
        else:
            sigma = fixed_sigma

        density += gaussian_2d_patch(sigma)

    return density


# ---------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------

DENSITY_GENERATORS = {
    # AnnotationType.DOT: _density_from_points,
    AnnotationType.DOT: density_from_points_csrnet_reference,
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
    - Smaller sigma → narrower blob (less spread), higher peak at the dot.
    - Larger sigma → wider blob (more spread), lower peak.
    Peak value at center = 1 / (2*pi*sigma^2); integral over plane = 1 per dot.

    geometry_adaptive:
      CSRNet-style geometry-adaptive kernels (Li et al., CVPR 2018, following [18]):
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