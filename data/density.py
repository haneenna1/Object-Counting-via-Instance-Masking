"""
Ground-truth density map generation from annotations.
Uses normalized 2D Gaussians so that the integral over the image approximates the object count.
"""

import numpy as np
from typing import List, Tuple, Union

from .annotation_types import AnnotationType


# ---------------------------------------------------------
# GAUSSIAN UTILITY
# ---------------------------------------------------------

def _gaussian_2d(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    sigma: Union[float, Tuple[float, float]],
    normalize: bool = True,
) -> np.ndarray:

    H, W = shape

    if isinstance(sigma, (int, float)):
        sigma_y = sigma_x = float(sigma)
    else:
        sigma_y, sigma_x = sigma

    cy, cx = center

    y = np.arange(H, dtype=np.float64) - cy
    x = np.arange(W, dtype=np.float64) - cx

    yy, xx = np.meshgrid(y, x, indexing="ij")

    g = np.exp(-(yy ** 2 / (2 * sigma_y ** 2) + xx ** 2 / (2 * sigma_x ** 2)))

    if normalize:
        g = g / (2 * np.pi * sigma_y * sigma_x)

    return g


# ---------------------------------------------------------
# DOT HANDLER
# ---------------------------------------------------------

def _density_from_points(shape, points, params):

    sigma = params["sigma"]

    H, W = shape
    density = np.zeros((H, W), dtype=np.float64)

    for (x, y) in points:
        density += _gaussian_2d(shape, (y, x), sigma, normalize=True)

    return density


# ---------------------------------------------------------
# BBOX HANDLER
# ---------------------------------------------------------

def _density_from_bboxes(shape, bboxes, params):

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

        density += _gaussian_2d(shape, (cy, cx), sigma, normalize=True)

    return density


# ---------------------------------------------------------
# SEGMENTATION HANDLER
# ---------------------------------------------------------

def _density_from_segmentations(shape, masks, params):

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

        density += _gaussian_2d(shape, (cy, cx), sigma, normalize=True)

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
) -> np.ndarray:

    """
    Generate ground-truth density map from annotations.
    """

    params = dict(
        sigma=sigma,
        sigma_scale_bbox=sigma_scale_bbox,
        sigma_from_seg_area=sigma_from_seg_area,
        fixed_sigma_seg=fixed_sigma_seg,
    )

    try:
        handler = DENSITY_GENERATORS[annotation_type]
    except KeyError:
        raise ValueError(f"Unsupported annotation_type: {annotation_type}")

    return handler(shape, annotations, params)