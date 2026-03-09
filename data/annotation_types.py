"""Annotation type enum for object counting datasets."""

from enum import Enum


class AnnotationType(str, Enum):
    """Supported annotation types for object counting."""

    DOT = "dot"           # One point per object (dotted annotation)
    BBOX = "bbox"          # Bounding box per object
    SEGMENTATION = "segmentation"  # Instance segmentation mask per object
