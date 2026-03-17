"""
Unified object counting dataset: loads images and produces density maps and instance masks
according to annotation type (dot, bbox, segmentation).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from .annotation_types import AnnotationType
from .density import generate_density
from .masking import generate_instance_mask


# Sentinel: use per-image default location <image_path>/../density_maps/<stem>_density.npy
DENSITY_MAP_DIR_AUTO = "auto"

# Scale density from [0, 1] to [0, DENSITY_SCALE] so the network sees larger targets and
# gets stronger gradients. Counts are recovered as density.sum() / DENSITY_SCALE.
# Training must use this same constant when converting density sums to counts.
DENSITY_SCALE = 255.0


# Type aliases for annotations
DotAnnotations = List[Tuple[float, float]]
BboxAnnotations = List[Tuple[float, float, float, float]]
SegmentationAnnotations = List[np.ndarray]



def _load_image(path: Union[str, Path]) -> torch.Tensor:
    """
    Load image as tensor (3,H,W) in [0,1].
    """
    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 1:  # grayscale
        img = img.repeat(3, 1, 1)
    return img

def _parse_annotation_type(value: Any) -> AnnotationType:
    if isinstance(value, AnnotationType):
        return value
    if isinstance(value, str):
        return AnnotationType(value.lower())
    raise ValueError(f"Cannot parse annotation type: {value}")


def _density_map_path_for_sample(
    item: Dict[str, Any],
    root: Optional[Path],
    density_map_dir: Union[Path, str],
) -> Path:
    """
    Path to the saved density map .npy file for a sample.
    If density_map_dir is DENSITY_MAP_DIR_AUTO, uses <image_path>/../density_maps/<stem>_density.npy.
    Otherwise uses density_map_dir / <stem>_density.npy.
    """
    rel = item["image_path"]
    stem = Path(rel).stem
    if density_map_dir == DENSITY_MAP_DIR_AUTO:
        full_image = (root or Path(".")) / rel
        # <image_path>/../density_maps/<stem>_density.npy
        out_dir = full_image.parent.parent / "density_maps"
        return out_dir / f"{stem}_density.npy"
    return Path(density_map_dir) / f"{stem}_density.npy"


class ObjectCountingDataset(Dataset):
    """
    Dataset that yields tensors:
        image  : (3,H,W)
        density: (1,H,W)
        mask   : (1,H,W)

    Each sample is a dict with:
        - "image_path"
        - "annotation_type": "dot" | "bbox" | "segmentation"
        - "annotations"
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        root: Optional[Union[str, Path]] = None,
        *,
        density_sigma: float = 4.0,
        density_sigma_scale_bbox: float = 0.25,
        density_sigma_from_seg_area: bool = True,
        density_fixed_sigma_seg: float = 4.0,
        mask_dot_box_size: Optional[Union[int, Tuple[int, int]]] = None,
        mask_dot_sigma_to_box: float = 2.0,
        mask_dot_sigma: float = 4.0,
        mask_object_ratio: Optional[float] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        keep_original_image: bool = False,
        density_map_dir: Optional[Union[str, Path]] = DENSITY_MAP_DIR_AUTO,
        density_scale: float = DENSITY_SCALE,
    ):
        self.samples = samples
        self.density_scale = density_scale
        self.root = Path(root) if root else None
        self.density_sigma = density_sigma
        self.density_sigma_scale_bbox = density_sigma_scale_bbox
        self.density_sigma_from_seg_area = density_sigma_from_seg_area
        self.density_fixed_sigma_seg = density_fixed_sigma_seg
        self.mask_dot_box_size = mask_dot_box_size
        self.mask_dot_sigma_to_box = mask_dot_sigma_to_box
        self.mask_dot_sigma = mask_dot_sigma
        self.mask_object_ratio = mask_object_ratio
        self.transform = transform
        self.keep_original_image = keep_original_image
        if density_map_dir is None:
            self.density_map_dir = None
        elif density_map_dir == DENSITY_MAP_DIR_AUTO:
            self.density_map_dir = DENSITY_MAP_DIR_AUTO
        else:
            self.density_map_dir = Path(density_map_dir)
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        item = self.samples[idx]

        image_path = item["image_path"]
        if self.root is not None:
            image_path = self.root / image_path
        else:
            image_path = Path(image_path)

        image = _load_image(image_path)
        _, H, W = image.shape
        shape = (H, W)

        ann_type = _parse_annotation_type(item["annotation_type"])
        annotations = item["annotations"]

        if self.density_map_dir is not None:
            density_path = _density_map_path_for_sample(
                item, self.root, self.density_map_dir
            )
            if density_path.exists():
                density = np.load(str(density_path)).astype(np.float32)
                density = torch.from_numpy(density).unsqueeze(0)
            else:
                density = generate_density(
                    shape,
                    ann_type,
                    annotations,
                    sigma=self.density_sigma,
                    sigma_scale_bbox=self.density_sigma_scale_bbox,
                    sigma_from_seg_area=self.density_sigma_from_seg_area,
                    fixed_sigma_seg=self.density_fixed_sigma_seg,
                )
                density = torch.from_numpy(density.astype(np.float32)).unsqueeze(0)
                density_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(density_path), density.squeeze(0).numpy())
        else:
            density = generate_density(
                shape,
                ann_type,
                annotations,
                sigma=self.density_sigma,
                sigma_scale_bbox=self.density_sigma_scale_bbox,
                sigma_from_seg_area=self.density_sigma_from_seg_area,
                fixed_sigma_seg=self.density_fixed_sigma_seg,
            )
            density = torch.from_numpy(density.astype(np.float32)).unsqueeze(0)

        # Scale density to [0, density_scale] so the network gets larger targets (stronger gradients).
        # Count = sum(density) / density_scale.
        density = density * self.density_scale
        count = density.sum().item() / self.density_scale

        mask = generate_instance_mask(
            shape,
            ann_type,
            annotations,
            dot_box_size=self.mask_dot_box_size,
            dot_sigma_to_box=self.mask_dot_sigma_to_box,
            dot_sigma=self.mask_dot_sigma,
            mask_object_ratio=self.mask_object_ratio,
        )

        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        out = {
            "image": image,
            "density": density,
            "mask": mask,
            "annotation_type": ann_type,
            "count": torch.tensor(count, dtype=torch.float32),
        }

        if self.transform is not None:
            out = self.transform(out)

        # Keep an unmasked copy for visualization/debugging.
        if self.keep_original_image:
            out["original_image"] = out["image"].clone()

        # mask: 1 = hide (object), 0 = show. (1,H,W) * (3,H,W) -> channel-wise broadcast
        out["image"] = out["image"] * (1.0 - out["mask"].clamp(0.0, 1.0))

        return out


def precompute_density_maps(
    dataset: ObjectCountingDataset,
    density_map_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Union[Path, str]:
    """
    Pre-compute and save density maps for all samples so that __getitem__ can load them.

    When density_map_dir is not passed, uses the dataset's density_map_dir (e.g. DENSITY_MAP_DIR_AUTO
    for per-image default: <image_path>/../density_maps/<stem>_density.npy).
    Returns the effective directory or DENSITY_MAP_DIR_AUTO when using per-image paths.
    """
    effective = (
        Path(density_map_dir) if density_map_dir is not None and density_map_dir != DENSITY_MAP_DIR_AUTO
        else (dataset.density_map_dir if dataset.density_map_dir is not None else DENSITY_MAP_DIR_AUTO)
    )
    root = dataset.root
    if effective != DENSITY_MAP_DIR_AUTO:
        effective.mkdir(parents=True, exist_ok=True)

    for item in dataset.samples:
        path = _density_map_path_for_sample(item, root, effective)
        if path.exists() and not force:
            continue
        image_path = item["image_path"]
        if root is not None:
            full_image_path = root / image_path
        else:
            full_image_path = Path(image_path)
        image = _load_image(full_image_path)
        _, H, W = image.shape
        shape = (H, W)
        ann_type = _parse_annotation_type(item["annotation_type"])
        annotations = item["annotations"]
        density = generate_density(
            shape,
            ann_type,
            annotations,
            sigma=dataset.density_sigma,
            sigma_scale_bbox=dataset.density_sigma_scale_bbox,
            sigma_from_seg_area=dataset.density_sigma_from_seg_area,
            fixed_sigma_seg=dataset.density_fixed_sigma_seg,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), density.astype(np.float32))
        print(f"Precomputed density map saved to {path}")

    return effective


def visualize_image_and_density(
    dataset: ObjectCountingDataset,
    image_name: str,
    *,
    figsize: Tuple[float, float] = (12, 5),
    density_cmap: str = "jet",
    show_count: bool = True,
    use_precomputed_density: bool = False,
    model: Optional[torch.nn.Module] = None,
) -> None:
    """
    Load an image by name, compute (or load) its ground-truth density map, and visualize.

    image_name: filename or stem (e.g. "image.jpg" or "image") to match against sample image_path.
    figsize: (width, height) for the figure.
    density_cmap: colormap for the density map (e.g. "jet", "viridis", "hot").
    show_count: if True, set title to include integral of density (object count).
    use_precomputed_density: when True, load density from dataset.density_map_dir instead of recomputing.
    model: optional torch.nn.Module. When provided, the model is evaluated on the image and its
           predicted density is shown alongside the ground-truth density.
    """
    name_stem = Path(image_name).stem
    sample = None
    for item in dataset.samples:
        if Path(item["image_path"]).stem == name_stem:
            sample = item
            break
    if sample is None:
        stems = [Path(s["image_path"]).stem for s in dataset.samples]
        hint = stems[:10] if len(stems) > 10 else stems
        raise ValueError(
            f"No sample found for image name {image_name!r} (stem: {name_stem!r}). "
            f"Available stems (sample): {hint}"
        )

    image_path = sample["image_path"]
    if dataset.root is not None:
        image_path = dataset.root / image_path
    else:
        image_path = Path(image_path)

    image = _load_image(image_path)
    _, H, W = image.shape
    shape = (H, W)
    ann_type = _parse_annotation_type(sample["annotation_type"])
    annotations = sample["annotations"]

    if use_precomputed_density:
        if dataset.density_map_dir is None:
            raise ValueError(
                "use_precomputed_density=True but dataset.density_map_dir is None. "
                "Either set density_map_dir when creating the dataset or disable use_precomputed_density."
            )
        density_path = _density_map_path_for_sample(
            sample,
            dataset.root,
            dataset.density_map_dir,
        )
        if not density_path.exists():
            raise FileNotFoundError(
                f"Precomputed density map not found at {density_path}. "
                "Run precompute_density_maps(...) first or set use_precomputed_density=False."
            )
        density = np.load(str(density_path)).astype(np.float32)
    else:
        density = generate_density(
            shape,
            ann_type,
            annotations,
            sigma=dataset.density_sigma,
            sigma_scale_bbox=dataset.density_sigma_scale_bbox,
            sigma_from_seg_area=dataset.density_sigma_from_seg_area,
            fixed_sigma_seg=dataset.density_fixed_sigma_seg,
        )

    count = float(np.sum(density))

    # Optional model prediction
    pred_arr: Optional[np.ndarray] = None
    pred_count: Optional[float] = None
    if model is not None:
        model.eval()
        with torch.no_grad():
            pred = model(image.unsqueeze(0))  # (1,1,H,W) or similar
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()
            if pred.ndim == 4 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            pred_arr = pred.numpy()
            pred_count = float(pred_arr.sum() / getattr(dataset, "density_scale", 1.0))

    has_dots = ann_type == AnnotationType.DOT and annotations

    # Layout:
    # - GT only, no dots: 2 panels (Image, GT density)
    # - Dots only: 3 panels (Image, Dots, GT density)
    # - GT + pred, no dots: 3 panels (Image, GT density, Pred density)
    # - Dots + pred: 4 panels (Image, Dots, GT density, Pred density)
    if has_dots and pred_arr is not None:
        fig, (ax_im, ax_dots, ax_den, ax_pred) = plt.subplots(
            1, 4, figsize=(figsize[0] * 2.0, figsize[1])
        )
    elif has_dots:
        fig, (ax_im, ax_dots, ax_den) = plt.subplots(
            1, 3, figsize=(figsize[0] * 1.5, figsize[1])
        )
        ax_pred = None  # type: ignore[assignment]
    elif pred_arr is not None:
        fig, (ax_im, ax_den, ax_pred) = plt.subplots(
            1, 3, figsize=(figsize[0] * 1.5, figsize[1])
        )
    else:
        fig, (ax_im, ax_den) = plt.subplots(1, 2, figsize=figsize)
        ax_dots = None  # type: ignore[assignment]
        ax_pred = None  # type: ignore[assignment]

    ax_im.imshow(image.permute(1, 2, 0).numpy())
    ax_im.set_title("Image")
    ax_im.axis("off")

    if has_dots:
        ax_dots.imshow(image.permute(1, 2, 0).numpy())
        xs = [p[0] for p in annotations]
        ys = [p[1] for p in annotations]
        ax_dots.scatter(xs, ys, c="lime", s=12, edgecolors="darkgreen", linewidths=0.5, zorder=5)
        ax_dots.set_title(f"Dots ({len(annotations)})")
        ax_dots.axis("off")

    im = ax_den.imshow(density, cmap=density_cmap)
    ax_den.set_title(f"GT density" + (f" (count ≈ {count:.1f})" if show_count else ""))
    ax_den.axis("off")
    plt.colorbar(im, ax=ax_den, fraction=0.046, pad=0.04)

    if pred_arr is not None and ax_pred is not None:
        im_pred = ax_pred.imshow(pred_arr, cmap=density_cmap)
        if show_count and pred_count is not None:
            ax_pred.set_title(f"Pred density (count ≈ {pred_count:.1f})")
        else:
            ax_pred.set_title("Pred density")
        ax_pred.axis("off")
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()