"""
Unified object counting dataset: loads images and produces density maps and instance masks
according to annotation type (dot, bbox, segmentation).
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from .annotation_types import AnnotationType
from .density import generate_density
from .masking import generate_instance_mask
from .transforms import crop_sample, horizontal_flip_transform


# Sentinel: use per-image default location <image_path>/../density_maps/<stem>_density.npy
DENSITY_MAP_DIR_AUTO = "auto"

# Ground-truth density maps are unscaled here: integral ≈ object count.
# Apply training-time scaling only in training/train.py (density_scale) for targets and pred→count.

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


class PatchAugmentedDataset(Dataset):
    """
    Expand each base sample into CSRNet-style quarter-sized training patches.

    For every image this wrapper yields:
    - 4 fixed quarter crops (top-left, top-right, bottom-left, bottom-right)
    - 5 additional random crops of the same size
    - optional horizontal mirrors of all crops
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        random_crops_per_image: int = 5,
        mirror: bool = True,
        seed: int = 0,
    ) -> None:
        if random_crops_per_image < 0:
            raise ValueError("random_crops_per_image must be >= 0")

        self.dataset = dataset
        self.random_crops_per_image = random_crops_per_image
        self.mirror = mirror
        self.seed = seed

        self.base_variants_per_image = 4 + self.random_crops_per_image
        self.variants_per_image = self.base_variants_per_image * (2 if self.mirror else 1)

    def __len__(self) -> int:
        return len(self.dataset) * self.variants_per_image

    def decode_variant_index(self, variant_idx: int) -> Tuple[int, bool]:
        if variant_idx < 0 or variant_idx >= self.variants_per_image:
            raise IndexError("CSRNetPatchAugmentedDataset variant index out of range")
        flip = self.mirror and variant_idx >= self.base_variants_per_image
        crop_variant = variant_idx % self.base_variants_per_image
        return crop_variant, flip

    def _crop_params(
        self,
        base_idx: int,
        crop_variant: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int]:
        crop_h = max(1, height // 2)
        crop_w = max(1, width // 2)
        max_top = max(height - crop_h, 0)
        max_left = max(width - crop_w, 0)

        if crop_variant < 4:
            quarter_offsets = (
                (0, 0),
                (0, max_left),
                (max_top, 0),
                (max_top, max_left),
            )
            top, left = quarter_offsets[crop_variant]
            return top, left, crop_h, crop_w

        random_crop_slot = crop_variant - 4
        rng_seed = self.seed + base_idx * 1009 + random_crop_slot * 9176
        rng = random.Random(rng_seed)
        top = rng.randint(0, max_top) if max_top > 0 else 0
        left = rng.randint(0, max_left) if max_left > 0 else 0
        return top, left, crop_h, crop_w

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("CSRNetPatchAugmentedDataset index out of range")

        base_idx, variant_idx = divmod(idx, self.variants_per_image) # base_idx: index of the original img, variant_idx: index of the variant of same img
        crop_variant, flip = self.decode_variant_index(variant_idx)

        sample = self.dataset[base_idx]
        _, height, width = sample["image"].shape
        top, left, crop_h, crop_w = self._crop_params(
            base_idx,
            crop_variant,
            height,
            width,
        )
        sample = crop_sample(sample, top=top, left=left, crop_h=crop_h, crop_w=crop_w)

        if flip:
            sample = horizontal_flip_transform(sample, p=1.0)

        return sample


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

    Instance masking (controlled by mask_object_ratio and mask_mode):
        mask_mode="robust"  — mask applied to both image AND density.
            Target is density of visible (unmasked) objects only.
            Acts as a robustness augmentation: the model learns to ignore
            missing regions and still estimate density for what it can see.
        mask_mode="inpaint" — mask applied to image only; density stays full.
            The model must learn to "hallucinate" / reconstruct the density
            of masked objects from surrounding context.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        root: Optional[Union[str, Path]] = None,
        *,
        density_sigma: float = 4.0,
        density_geometry_adaptive: bool = False,
        density_beta: float = 0.3,
        density_k: int = 3,
        density_min_sigma: float = 4.0,
        density_sigma_scale_bbox: float = 0.25,
        density_sigma_from_seg_area: bool = True,
        density_fixed_sigma_seg: float = 4.0,
        mask_dot_box_size: Optional[Union[int, Tuple[int, int]]] = None,
        mask_dot_sigma_to_box: float = 2.0,
        mask_dot_sigma: float = 4.0,
        mask_object_ratio: Optional[float] = None,
        mask_mode: str = "inpaint",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        keep_original_image: bool = False,
        density_map_dir: Optional[Union[str, Path]] = DENSITY_MAP_DIR_AUTO,
    ):
        if mask_mode not in ("robust", "inpaint"):
            raise ValueError(
                f"mask_mode must be 'robust' or 'inpaint', got {mask_mode!r}"
            )
        self.samples = samples
        self.root = Path(root) if root else None
        self.density_sigma = density_sigma
        self.density_geometry_adaptive = density_geometry_adaptive
        self.density_beta = density_beta
        self.density_k = density_k
        self.density_min_sigma = density_min_sigma
        self.density_sigma_scale_bbox = density_sigma_scale_bbox
        self.density_sigma_from_seg_area = density_sigma_from_seg_area
        self.density_fixed_sigma_seg = density_fixed_sigma_seg
        self.mask_dot_box_size = mask_dot_box_size
        self.mask_dot_sigma_to_box = mask_dot_sigma_to_box
        self.mask_dot_sigma = mask_dot_sigma
        self.mask_object_ratio = mask_object_ratio
        self.mask_mode = mask_mode
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
                    geometry_adaptive=self.density_geometry_adaptive,
                    beta=self.density_beta,
                    k=self.density_k,
                    min_sigma=self.density_min_sigma,
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
                geometry_adaptive=self.density_geometry_adaptive,
                beta=self.density_beta,
                k=self.density_k,
                min_sigma=self.density_min_sigma,
                sigma_scale_bbox=self.density_sigma_scale_bbox,
                sigma_from_seg_area=self.density_sigma_from_seg_area,
                fixed_sigma_seg=self.density_fixed_sigma_seg,
            )
            density = torch.from_numpy(density.astype(np.float32)).unsqueeze(0)

        # Raw density: sum ≈ object count (see data/density.py Gaussians).
        count = float(density.sum().item())

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

        # Instance masking: binary_mask 1 = hide (object), 0 = show.
        binary_mask = out["mask"].clamp(0.0, 1.0)
        out["image"] = out["image"] * (1.0 - binary_mask)

        if self.mask_mode == "robust":
            # Also mask density — model learns density of visible objects only.
            out["density"] = out["density"] * (1.0 - binary_mask)
            out["count"] = torch.tensor(
                float(out["density"].sum().item()), dtype=torch.float32
            )
        # "inpaint": density stays full — model must hallucinate masked density.

        return out


def precompute_density_maps(
    dataset: ObjectCountingDataset,
    density_map_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    *,
    density_geometry_adaptive: Optional[bool] = None,
    density_beta: Optional[float] = None,
    density_k: Optional[int] = None,
    density_min_sigma: Optional[float] = None,
) -> Union[Path, str]:
    """
    Pre-compute and save density maps for all samples so that __getitem__ can load them.

    When density_map_dir is not passed, uses the dataset's density_map_dir (e.g. DENSITY_MAP_DIR_AUTO
    for per-image default: <image_path>/../density_maps/<stem>_density.npy).
    Returns the effective directory or DENSITY_MAP_DIR_AUTO when using per-image paths.

    If any of density_* arguments are provided (not None), they override the dataset's
    corresponding density generation settings for this precompute call.
    """
    effective = (
        Path(density_map_dir) if density_map_dir is not None and density_map_dir != DENSITY_MAP_DIR_AUTO
        else (dataset.density_map_dir if dataset.density_map_dir is not None else DENSITY_MAP_DIR_AUTO)
    )
    root = dataset.root
    if effective != DENSITY_MAP_DIR_AUTO:
        effective.mkdir(parents=True, exist_ok=True)

    # Allow one-off overrides for adaptive kernels during precompute.
    geometry_adaptive = (
        dataset.density_geometry_adaptive
        if density_geometry_adaptive is None
        else density_geometry_adaptive
    )
    beta = dataset.density_beta if density_beta is None else density_beta
    k = dataset.density_k if density_k is None else density_k
    min_sigma = dataset.density_min_sigma if density_min_sigma is None else density_min_sigma

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
            geometry_adaptive=geometry_adaptive,
            beta=beta,
            k=k,
            min_sigma=min_sigma,
            sigma_scale_bbox=dataset.density_sigma_scale_bbox,
            sigma_from_seg_area=dataset.density_sigma_from_seg_area,
            fixed_sigma_seg=dataset.density_fixed_sigma_seg,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), density.astype(np.float32))
        print(f"Precomputed density map saved to {path}")

    return effective


def visualize_csrnet_patch_augmented_dataset(
    dataset: PatchAugmentedDataset,
    base_idx: int = 0,
    *,
    include_mirrored: bool = False,
    figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
    save_path: Optional[Union[str, Path]] = "csrnet_patch_visualization.png",
    show: bool = True,
) -> None:
    """
    Visualize one original image together with all CSRNet quarter-sized patches.

    The first panel shows the original image with crop boxes overlaid.
    The remaining panels show the cropped patches in the same order used by
    PatchAugmentedDataset.
    """
    if base_idx < 0 or base_idx >= len(dataset.dataset):
        raise IndexError("base_idx out of range for base dataset")
    if not isinstance(dataset.dataset, ObjectCountingDataset):
        raise TypeError(
            "visualize_csrnet_patch_augmented_dataset expects "
            "CSRNetPatchAugmentedDataset wrapping ObjectCountingDataset."
        )

    base_dataset = dataset.dataset
    item = base_dataset.samples[base_idx]
    image_path = item["image_path"]
    if base_dataset.root is not None:
        image_path = base_dataset.root / image_path
    else:
        image_path = Path(image_path)

    image = _load_image(image_path)
    _, height, width = image.shape
    image_np = image.permute(1, 2, 0).cpu().numpy()

    patch_panels: List[Tuple[str, np.ndarray, Tuple[int, int, int, int], bool]] = []
    total_variants = dataset.variants_per_image if include_mirrored and dataset.mirror else dataset.base_variants_per_image

    for variant_idx in range(total_variants):
        crop_variant, flip = dataset.decode_variant_index(variant_idx)
        top, left, crop_h, crop_w = dataset._crop_params(
            base_idx,
            crop_variant,
            height,
            width,
        )
        patch = image[:, top : top + crop_h, left : left + crop_w]
        if flip:
            patch = torch.flip(patch, dims=[2])

        if crop_variant < 4:
            label = f"Quarter {crop_variant + 1}"
        else:
            label = f"Random {crop_variant - 3}"
        if flip:
            label += " (mirrored)"

        patch_panels.append(
            (
                label,
                patch.permute(1, 2, 0).cpu().numpy(),
                (top, left, crop_h, crop_w),
                flip,
            )
        )

    total_panels = 1 + len(patch_panels)
    cols = min(5, total_panels)
    rows = (total_panels + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)
    axes_flat = axes.ravel()

    ax_original = axes_flat[0]
    ax_original.imshow(image_np)
    ax_original.set_title(f"Original image #{base_idx}")
    ax_original.axis("off")

    box_colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    random_color = "tab:orange"
    for crop_idx, (label, _, (top, left, crop_h, crop_w), _) in enumerate(patch_panels):
        color = box_colors[crop_idx] if crop_idx < 4 else random_color
        rect = Rectangle(
            (left, top),
            crop_w,
            crop_h,
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle="--" if "mirrored" in label else "-",
        )
        ax_original.add_patch(rect)
        ax_original.text(
            left + 4,
            top + 18,
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
        )

    for panel_idx, (label, patch_np, _, _) in enumerate(patch_panels, start=1):
        ax = axes_flat[panel_idx]
        ax.imshow(patch_np)
        ax.set_title(label)
        ax.axis("off")

    for ax in axes_flat[total_panels:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def visualize_image_and_density(
    dataset: ObjectCountingDataset,
    image_name: str,
    *,
    figsize: Tuple[float, float] = (12, 5),
    density_cmap: str = "jet",
    show_count: bool = True,
    use_precomputed_density: bool = False,
    adaptive: bool = True,
    model: Optional[torch.nn.Module] = None,
    pred_density_scale: float = 1.0,
    save_path: Optional[Union[str, Path]] = "visualization.png",
    show: bool = False,
    device: Optional[torch.device] = 'cuda'
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
    pred_density_scale: divide predicted map sum by this to get count (match train(..., density_scale=...)).
    """
    # image_name can be a stem, filename, or a relative path such as
    # "part_A/train_data/images/IMG_1.jpg".
    image_name_path = Path(image_name)
    name_stem = image_name_path.stem

    sample: Optional[Dict[str, Any]] = None

    print(f"visualization: image name={image_name_path} precomputed density={use_precomputed_density} adaptive={adaptive} beta={dataset.density_beta} k={dataset.density_k} min_sigma={dataset.density_min_sigma}")

    # 1) Prefer exact relative-path match against item["image_path"].
    for item in dataset.samples:
        item_path = Path(item["image_path"])
        if image_name_path == item_path:
            sample = item
            break

    # 2) Fallback: match by filename (or stem) if no exact path match was found.
    if sample is None:
        for item in dataset.samples:
            item_path = Path(item["image_path"])
            if image_name_path.name == item_path.name:
                sample = item
                break

    if sample is None:
        paths = [s["image_path"] for s in dataset.samples]
        hint = paths[:10] if len(paths) > 10 else paths
        raise ValueError(
            f"No sample found for image name {image_name!r} (stem: {name_stem!r}). "
            f"Try using a relative path like one of: {hint}"
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
            geometry_adaptive=adaptive,
            beta=dataset.density_beta,
            k=dataset.density_k,
            min_sigma=dataset.density_min_sigma,
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
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            pred = model(image.unsqueeze(0))  # (1,1,H,W) or similar
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()
            if pred.ndim == 4 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            pred_count = float(pred.sum().item() / pred_density_scale)

            pred_h, pred_w = pred.shape[-2], pred.shape[-1]
            H_gt, W_gt = density.shape[:2]
            if pred_h != H_gt or pred_w != W_gt:
                density_t = torch.from_numpy(density).unsqueeze(0).unsqueeze(0).float()
                density_t = F.interpolate(
                    density_t, size=(pred_h, pred_w),
                    mode="bilinear", align_corners=False,
                )
                spatial_scale = (H_gt * W_gt) / (pred_h * pred_w)
                density_t = density_t * spatial_scale
                density = density_t.squeeze().numpy()
                count = float(density.sum())
            pred_arr = pred.squeeze().numpy()

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

    ax_im.imshow(image.permute(1, 2, 0).to('cpu').numpy())
    ax_im.set_title("Image")
    ax_im.axis("off")

    if has_dots:
        ax_dots.imshow(image.permute(1, 2, 0).to('cpu').numpy())
        xs = [p[0] for p in annotations]
        ys = [p[1] for p in annotations]
        ax_dots.scatter(xs, ys, c="lime", s=12, edgecolors="darkgreen", linewidths=0.5, zorder=5)
        ax_dots.set_title(f"Dots ({len(annotations)})")
        ax_dots.axis("off")

    im = ax_den.imshow(density, cmap=density_cmap)
    ax_den.set_title(f"GT density" + (f"\n(count ≈ {count:.1f})" + f"\nadaptive={adaptive} " + f"beta={dataset.density_beta} " + f"\nk={dataset.density_k} " + f"min_sigma={dataset.density_min_sigma} "  if show_count else ""))
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

    # Save to file when requested (useful on headless/SSH setups).
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)

    # Optionally show the figure in environments with a GUI backend.
    if show:
        plt.show()

    plt.close()