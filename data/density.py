"""
Ground-truth density map generation from annotations.
Uses normalized 2D Gaussians so that the integral over the image approximates the object count.
"""

import numpy as np
from typing import List, Tuple, Union

from .annotation_types import AnnotationType


def _gaussian_2d(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    sigma: Union[float, Tuple[float, float]],
    normalize: bool = True,
) -> np.ndarray:
    """
    Draw a single normalized 2D Gaussian on a grid.

    Args:
        shape: (H, W) output shape
        center: (x, y) or (y, x) - we use (y, x) for array indexing
        sigma: scalar or (sigma_y, sigma_x) for isotropic or axis-aligned Gaussian
        normalize: if True, kernel sums to 1 (so integral over image ~ count contribution)

    Returns:
        Density map (H, W), float64
    """
    H, W = shape
    if isinstance(sigma, (int, float)):
        sigma_y = sigma_x = float(sigma)
    else:
        sigma_y, sigma_x = sigma

    # Grid in pixel coordinates (row = y, col = x)
    cy, cx = center
    y = np.arange(H, dtype=np.float64) - cy
    x = np.arange(W, dtype=np.float64) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")

    g = np.exp(-(yy ** 2 / (2 * sigma_y ** 2) + xx ** 2 / (2 * sigma_x ** 2)))
    if normalize:
        g = g / (2 * np.pi * sigma_y * sigma_x)
    return g


def _density_from_points(
    shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Density as sum of normalized Gaussians at each point (NIPS 2010 style).
    points: list of (x, y) in pixel coordinates.
    """
    H, W = shape
    density = np.zeros((H, W), dtype=np.float64)
    for (x, y) in points:
        # center in (row, col) = (y, x)
        density += _gaussian_2d(shape, (y, x), sigma, normalize=True)
    return density


def _density_from_bboxes(
    shape: Tuple[int, int],
    bboxes: List[Tuple[float, float, float, float]],
    sigma_scale: float = 0.25,
) -> np.ndarray:
    """
    Density as sum of normalized Gaussians at bbox centers.
    Sigma is derived from bbox size (geometry-adaptive).
    bboxes: list of (x1, y1, x2, y2).
    """
    H, W = shape
    density = np.zeros((H, W), dtype=np.float64)
    for (x1, y1, x2, y2) in bboxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        # sigma proportional to object size (e.g. half of mean side)
        sigma = sigma_scale * (w + h) / 2.0
        sigma = max(sigma, 1.0)
        density += _gaussian_2d(shape, (cy, cx), sigma, normalize=True)
    return density


def _density_from_segmentations(
    shape: Tuple[int, int],
    masks: List[np.ndarray],
    sigma_from_area: bool = True,
    fixed_sigma: float = 4.0,
) -> np.ndarray:
    """
    Density from instance segmentation masks.
    Each instance contributes a Gaussian centered at the segment centroid;
    sigma is derived from segment area or fixed.
    masks: list of (H, W) binary arrays (same H, W as shape).
    """
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
        # Centroid (row, col) = (y, x)
        rows, cols = np.where(binary > 0)
        cy = rows.mean()
        cx = cols.mean()
        if sigma_from_area:
            # sigma ~ sqrt(area) so Gaussian "covers" the instance roughly
            sigma = np.sqrt(area) * 0.5
            sigma = max(sigma, 1.0)
        else:
            sigma = fixed_sigma
        density += _gaussian_2d(shape, (cy, cx), sigma, normalize=True)
    return density


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

    Args:
        shape: (H, W) of the image.
        annotation_type: One of AnnotationType.DOT, BBOX, SEGMENTATION.
        annotations:
            - For DOT: list of (x, y) per object.
            - For BBOX: list of (x1, y1, x2, y2) per object.
            - For SEGMENTATION: list of (H, W) binary masks per object.
        sigma: For DOT: Gaussian sigma in pixels (default 4).
        sigma_scale_bbox: For BBOX: scale factor to derive sigma from bbox size.
        sigma_from_seg_area: For SEGMENTATION: if True, sigma from segment area.
        fixed_sigma_seg: For SEGMENTATION: fixed sigma when sigma_from_seg_area is False.

    Returns:
        density: (H, W) float64 array; sum over region approximates count.
    """
    if annotation_type == AnnotationType.DOT:
        return _density_from_points(shape, annotations, sigma=sigma)
    if annotation_type == AnnotationType.BBOX:
        return _density_from_bboxes(shape, annotations, sigma_scale=sigma_scale_bbox)
    if annotation_type == AnnotationType.SEGMENTATION:
        return _density_from_segmentations(
            shape,
            annotations,
            sigma_from_area=sigma_from_seg_area,
            fixed_sigma=fixed_sigma_seg,
        )
    raise ValueError(f"Unsupported annotation_type: {annotation_type}")
