"""
Data loading for object counting: density map generation and instance masking.
Supports dotted, bbox, and segmentation annotations.
"""

from .annotation_types import AnnotationType
from .density import density_from_points_csrnet_reference, generate_density
from .masking import generate_instance_mask
from .dataset import (
    PatchAugmentedDataset,
    ObjectCountingDataset,
    visualize_csrnet_patch_augmented_dataset,
)
from .shanghaitech import (
    build_shanghaitech_samples,
    load_shanghaitech_dataset,
    ShanghaiTechDataset,
)
from .fsc147 import (
    build_fsc147_samples,
    load_fsc147_dataset,
    FSC147Dataset,
)

__all__ = [
    "AnnotationType",
    "generate_density",
    "density_from_points_csrnet_reference",
    "generate_instance_mask",
    "PatchAugmentedDataset",
    "ObjectCountingDataset",
    "visualize_csrnet_patch_augmented_dataset",
    "build_shanghaitech_samples",
    "load_shanghaitech_dataset",
    "ShanghaiTechDataset",
    "build_fsc147_samples",
    "load_fsc147_dataset",
    "FSC147Dataset",
]