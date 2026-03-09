"""
Data loading for object counting: density map generation and instance masking.
Supports dotted, bbox, and segmentation annotations.
"""

from .annotation_types import AnnotationType
from .density import generate_density
from .masking import generate_instance_mask
from .dataset import ObjectCountingDataset
from .shanghaitech import (
    build_shanghaitech_samples,
    load_shanghaitech_dataset,
    ShanghaiTechDataset,
)

__all__ = [
    "AnnotationType",
    "generate_density",
    "generate_instance_mask",
    "ObjectCountingDataset",
    "build_shanghaitech_samples",
    "load_shanghaitech_dataset",
    "ShanghaiTechDataset",
]

try:
    from .dataset import TorchObjectCountingDataset
    __all__.append("TorchObjectCountingDataset")
except ImportError:
    pass
