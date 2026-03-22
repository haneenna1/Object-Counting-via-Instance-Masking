"""
Ground-truth density map generation from annotations.
Uses normalized 2D Gaussians so that the integral over the image approximates the object count.
"""

import numpy as np
from typing import List, Tuple, Union
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

    # -----------------------------------------------------
    # Compute adaptive sigmas (kNN)
    # -----------------------------------------------------
    if geometry_adaptive and len(points) > k:
        tree = KDTree(pts)

        # k+1 because first neighbor is the point itself
        dists, _ = tree.query(pts, k=k + 1)
        dists = dists[:, 1:]  # remove self-distance

        mean_dists = dists.mean(axis=1)

        sigmas = beta * mean_dists
        # sigmas = np.clip(sigmas, min_sigma, max_sigma)

    else:
        sigmas = np.full(len(points), sigma, dtype=np.float64)
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

# ---------------------------------------------------------
# BBOX HANDLER
# ---------------------------------------------------------

def _density_from_bboxes(shape, bboxes, params, **kwargs):

    sigma_scale = params["sigma_scale_bbox"]

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

def _density_from_segmentations(shape, masks, params, **kwargs):

    sigma_from_area = params["sigma_from_seg_area"]
    fixed_sigma = params["fixed_sigma_seg"]

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