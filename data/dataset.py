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
from .density import generate_density, sum_dot_gaussians_for_indices
from .masking import generate_instance_mask
from .transforms import crop_sample, horizontal_flip_transform


# Sentinel: use per-image default location <image_path>/../density_maps/<stem>_density.npy
DENSITY_MAP_DIR_AUTO = "auto"


def _density_stats_text(arr: np.ndarray) -> str:
    """Multi-line summary for density panels (integral, spread, scale)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return "(empty)"
    return (
        f"Σ (integral) = {float(a.sum()):.6g}\n"
        f"min = {float(a.min()):.6g}\n"
        f"max = {float(a.max()):.6g}\n"
        f"mean = {float(a.mean()):.6g}\n"
        f"std = {float(a.std()):.6g}"
    )


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
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        if random_crops_per_image < 0:
            raise ValueError("random_crops_per_image must be >= 0")

        self.dataset = dataset
        self.random_crops_per_image = random_crops_per_image
        self.mirror = mirror
        self.seed = seed
        self.transform = transform
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
        # rng_seed = self.seed + base_idx * 1009 + random_crop_slot * 9176
        # rng = random.Random(rng_seed)
        rng = random.Random()

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

        if self.transform is not None:
            sample = self.transform(sample)

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
        mask_dot_box_aspect: (h, w) integer ratio per dot, e.g. (2, 1) → height = 2× width;
            ignored if mask_dot_box_size is a (height, width) tuple in pixels.
        mask_mode="robust"  — mask applied to both image AND density.
            Target is density of visible (unmasked) objects only.
            Acts as a robustness augmentation: the model learns to ignore
            missing regions and still estimate density for what it can see.
        mask_mode="inpaint" — mask applied to image only; density stays full.
            The model must learn to "hallucinate" / reconstruct the density
            of masked objects from surrounding context.
        mask_dot_style (dot annotations only):
            ``"box"`` — rectangular mask; robust mode uses density * (1 - mask).
            ``"gaussian"`` — image mask uses ~3σ disks from CSRNet-style sigmas;
            robust mode subtracts each masked head's GT Gaussian (matches generate_density),
            instead of zeroing a rectangle in the density map.
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
        mask_dot_box_aspect: Tuple[int, int] = (1, 1),
        mask_dot_sigma_to_box: float = 2.0,
        mask_dot_sigma: float = 4.0,
        mask_dot_geometry_adaptive: Optional[bool] = None,
        mask_dot_geometry_beta: Optional[float] = None,
        mask_dot_geometry_k: Optional[int] = None,
        mask_dot_geometry_min_sigma: Optional[float] = None,
        mask_dot_geometry_max_sigma: float = 15.0,
        mask_object_ratio: Optional[float] = None,
        mask_mode: str = "inpaint",
        mask_dot_style: str = "box",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        keep_original_image: bool = False,
        density_map_dir: Optional[Union[str, Path]] = DENSITY_MAP_DIR_AUTO,
    ):
        if mask_mode not in ("robust", "inpaint"):
            raise ValueError(
                f"mask_mode must be 'robust' or 'inpaint', got {mask_mode!r}"
            )
        if mask_dot_style not in ("box", "gaussian"):
            raise ValueError(
                f"mask_dot_style must be 'box' or 'gaussian', got {mask_dot_style!r}"
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
        self.mask_dot_box_aspect = mask_dot_box_aspect
        self.mask_dot_sigma_to_box = mask_dot_sigma_to_box
        self.mask_dot_sigma = mask_dot_sigma
        self.mask_dot_geometry_adaptive = (
            density_geometry_adaptive
            if mask_dot_geometry_adaptive is None
            else mask_dot_geometry_adaptive
        )
        self.mask_dot_geometry_beta = (
            density_beta if mask_dot_geometry_beta is None else mask_dot_geometry_beta
        )
        self.mask_dot_geometry_k = (
            density_k if mask_dot_geometry_k is None else mask_dot_geometry_k
        )
        self.mask_dot_geometry_min_sigma = (
            density_min_sigma
            if mask_dot_geometry_min_sigma is None
            else mask_dot_geometry_min_sigma
        )
        self.mask_dot_geometry_max_sigma = mask_dot_geometry_max_sigma
        self.mask_object_ratio = mask_object_ratio
        self.mask_mode = mask_mode
        self.mask_dot_style = mask_dot_style
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

        need_masked_idx = (
            self.mask_mode == "robust"
            and ann_type == AnnotationType.DOT
            and self.mask_dot_style == "gaussian"
            and self.mask_object_ratio not in (None, 0)
        )
        mask, masked_idx = generate_instance_mask(
            shape,
            ann_type,
            annotations,
            dot_box_size=self.mask_dot_box_size,
            dot_box_aspect=self.mask_dot_box_aspect,
            dot_sigma_to_box=self.mask_dot_sigma_to_box,
            dot_sigma=self.mask_dot_sigma,
            dot_geometry_adaptive=self.mask_dot_geometry_adaptive,
            dot_geometry_beta=self.mask_dot_geometry_beta,
            dot_geometry_k=self.mask_dot_geometry_k,
            dot_geometry_min_sigma=self.mask_dot_geometry_min_sigma,
            dot_geometry_max_sigma=self.mask_dot_geometry_max_sigma,
            mask_object_ratio=self.mask_object_ratio,
            dot_mask_style=self.mask_dot_style,
            return_masked_indices=need_masked_idx,
        )

        # Gaussian subtract must stay here (before ``out`` / ``transform``): it uses this
        # sample's ``shape`` and ``annotations`` in full image space. The robust branch
        # below runs after ``transform``; moving subtract there would break when the
        # transform resizes/crops (e.g. ViT dict transform on the dataset).
        gaussian_density_subtract = (
            self.mask_mode == "robust"
            and ann_type == AnnotationType.DOT
            and self.mask_dot_style == "gaussian"
            and masked_idx is not None
            and masked_idx.size > 0
        )
        if gaussian_density_subtract:
            d_np = density.squeeze(0).numpy().astype(np.float32)
            removed = sum_dot_gaussians_for_indices(
                shape, list(annotations), masked_idx
            )
            d_np = np.maximum(d_np - removed.astype(np.float32), 0.0)
            density = torch.from_numpy(d_np).unsqueeze(0)

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
            # Also adjust density — model learns density of visible objects only.
            if gaussian_density_subtract:
                out["count"] = torch.tensor(
                    float(out["density"].sum().item()), dtype=torch.float32
                )
            else:
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
    density_cmap: str = "jet",
    save_path: Optional[Union[str, Path]] = "csrnet_patch_visualization.png",
    show: bool = True,
) -> None:
    """
    Visualize one original image together with all CSRNet quarter-sized patches.

    Calls ``ObjectCountingDataset.__getitem__`` for the base image, then crops
    the result so panels reflect the actual training data (masking, density, etc.).

    The first column shows the full-size processed image (image / masked / density).
    Each subsequent column shows one patch variant.
    """
    if base_idx < 0 or base_idx >= len(dataset.dataset):
        raise IndexError("base_idx out of range for base dataset")
    if not isinstance(dataset.dataset, ObjectCountingDataset):
        raise TypeError(
            "visualize_csrnet_patch_augmented_dataset expects "
            "PatchAugmentedDataset wrapping ObjectCountingDataset."
        )

    base_dataset = dataset.dataset
    base_sample = base_dataset[base_idx]

    masked_image = base_sample["image"]           # (3,H,W) — already masked by __getitem__
    density = base_sample["density"]              # (1,H,W)
    count = float(base_sample["count"].item())
    has_original = "original_image" in base_sample
    original_image = base_sample["original_image"] if has_original else masked_image
    _, height, width = masked_image.shape

    def _to_np(t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu()
        if t.min() < -0.01 or t.max() > 1.01:
            t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
        return t.permute(1, 2, 0).numpy()

    total_variants = dataset.variants_per_image if include_mirrored and dataset.mirror else dataset.base_variants_per_image
    n_rows = 3 if has_original else 2  # [original], masked, density
    n_cols = 1 + total_variants        # full image + patches
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    row = 0
    if has_original:
        axes[row, 0].imshow(_to_np(original_image))
        axes[row, 0].set_title(f"Original #{base_idx} ({width}×{height})")
        axes[row, 0].axis("off")
        row += 1

    axes[row, 0].imshow(_to_np(masked_image))
    axes[row, 0].set_title(f"Masked (ratio={base_dataset.mask_object_ratio})")
    axes[row, 0].axis("off")
    crop_box_row = row

    den_np = density.squeeze(0).detach().cpu().numpy()
    den_row = row + 1
    im0 = axes[den_row, 0].imshow(den_np, cmap=density_cmap)
    axes[den_row, 0].set_title(f"Density (count≈{count:.1f})")
    axes[den_row, 0].axis("off")
    plt.colorbar(im0, ax=axes[den_row, 0], fraction=0.046, pad=0.04)

    box_colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    random_color = "tab:orange"

    for variant_idx in range(total_variants):
        col = variant_idx + 1
        crop_variant, flip = dataset.decode_variant_index(variant_idx)
        top, left, crop_h, crop_w = dataset._crop_params(
            base_idx, crop_variant, height, width,
        )

        if crop_variant < 4:
            label = f"Q{crop_variant + 1}"
        else:
            label = f"R{crop_variant - 3}"
        if flip:
            label += " (flip)"

        color = box_colors[crop_variant] if crop_variant < 4 else random_color
        rect = Rectangle(
            (left, top), crop_w, crop_h,
            fill=False, edgecolor=color, linewidth=2,
            linestyle="--" if flip else "-",
        )
        axes[crop_box_row, 0].add_patch(rect)

        def _crop_and_flip(t: torch.Tensor) -> torch.Tensor:
            c = t[:, top : top + crop_h, left : left + crop_w]
            if flip:
                c = torch.flip(c, dims=[2])
            return c

        r = 0
        if has_original:
            patch_orig = _crop_and_flip(original_image)
            axes[r, col].imshow(_to_np(patch_orig))
            axes[r, col].set_title(label)
            axes[r, col].axis("off")
            r += 1

        patch_masked = _crop_and_flip(masked_image)
        axes[r, col].imshow(_to_np(patch_masked))
        axes[r, col].set_title(label)
        axes[r, col].axis("off")

        patch_den = _crop_and_flip(density).squeeze(0).detach().cpu().numpy()
        patch_count = float(patch_den.sum())
        r += 1
        im_d = axes[r, col].imshow(patch_den, cmap=density_cmap)
        axes[r, col].set_title(f"{label}\ncount≈{patch_count:.1f}")
        axes[r, col].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def _object_counting_base_for_visualize(
    dataset: Union[ObjectCountingDataset, PatchAugmentedDataset],
) -> Tuple[ObjectCountingDataset, Optional[PatchAugmentedDataset]]:
    """Return the inner ObjectCountingDataset and optional PatchAugmentedDataset wrapper."""
    if isinstance(dataset, PatchAugmentedDataset):
        inner = dataset.dataset
        if not isinstance(inner, ObjectCountingDataset):
            raise TypeError(
                "visualize_image_and_density: PatchAugmentedDataset must wrap ObjectCountingDataset, "
                f"got {type(inner).__name__}."
            )
        return inner, dataset
    return dataset, None


def visualize_image_and_density(
    dataset: Union[ObjectCountingDataset, PatchAugmentedDataset],
    image_name: Optional[str] = None,
    *,
    index: Optional[int] = None,
    patch_variant: int = 0,
    figsize: Tuple[float, float] = (12, 5),
    density_cmap: str = "jet",
    use_precomputed_density: bool = False,
    adaptive: bool = True,
    use_dataset_item: bool = True,
    model: Optional[torch.nn.Module] = None,
    pred_density_scale: float = 1.0,
    save_dir: Optional[Union[str, Path]] = "vis/",
    device: Optional[torch.device] = 'cuda'
) -> None:
    """
    Visualize one sample: either by ``index`` into ``dataset`` or by ``image_name`` lookup.

    Pass exactly one of:
      - ``index``: integer in ``[0, len(dataset))`` (same row as ``dataset[index]``). For a
        ``PatchAugmentedDataset``, this is the **flattened** patch index (``base_idx * variants_per_image + variant``).
      - ``image_name``: filename, stem, or relative path matched against ``sample["image_path"]`` on the
        **base** ``ObjectCountingDataset``. If ``dataset`` is ``PatchAugmentedDataset``, use ``patch_variant``
        in ``[0, variants_per_image)`` to choose which crop/mirror (default ``0``).

    patch_variant: only used with ``image_name=...`` when ``dataset`` is ``PatchAugmentedDataset``; ignored otherwise.

    figsize: (width, height) for the figure.
    density_cmap: colormap for the density map (e.g. "jet", "viridis", "hot").
    use_precomputed_density: when True, load density from dataset.density_map_dir instead of recomputing.
      Ignored when use_dataset_item=True (the dataset __getitem__ path already loads or generates density).
    use_dataset_item: when True (default), call ``dataset[flat_idx]`` so the mask, masked image, and
      target density match the dataloader exactly (including random subsampling of instances).
      Set False only if you want an independent mask/density pass (e.g. legacy comparisons).
      If ``transform`` resizes (e.g. ViT 224×224), the first panel still shows the on-disk image at native
      resolution; masked image, GT density, and model input use the transformed spatial size. Normalized
      images (e.g. timm mean/std 0.5) are denormalized for display on the masked panel only.
    adaptive: when use_dataset_item=False, passed to generate_density for the GT map; when True and
      use_dataset_item=True, the figure subtitle still reflects dataset.density_geometry_adaptive.
    model: optional torch.nn.Module. When provided, the model is evaluated on the image and its
           predicted density is shown alongside the ground-truth density.
    pred_density_scale: divide predicted map sum by this to get count (match train(..., density_scale=...)).
    """
    if (index is not None) == (image_name is not None):
        raise ValueError("Pass exactly one of index=... or image_name=..., not both and not neither.")

    base_ds, patch_ds = _object_counting_base_for_visualize(dataset)

    sample: Optional[Dict[str, Any]] = None
    base_sample_idx: Optional[int] = None
    flat_idx: Optional[int] = None
    image_name_path: Path

    if index is not None:
        n = len(dataset)
        if index < 0 or index >= n:
            raise IndexError(f"index {index} out of range for dataset of length {n}")
        flat_idx = index
        if patch_ds is not None:
            base_sample_idx = flat_idx // patch_ds.variants_per_image
        else:
            base_sample_idx = flat_idx
        sample = base_ds.samples[base_sample_idx]
        image_name_path = Path(sample["image_path"])
        name_stem = image_name_path.stem
    else:
        assert image_name is not None
        image_name_path = Path(image_name)
        name_stem = image_name_path.stem

        # 1) Prefer exact relative-path match against item["image_path"].
        for i, item in enumerate(base_ds.samples):
            item_path = Path(item["image_path"])
            if image_name_path == item_path:
                sample = item
                base_sample_idx = i
                break

        # 2) Fallback: match by filename if no exact path match was found.
        if sample is None:
            for i, item in enumerate(base_ds.samples):
                item_path = Path(item["image_path"])
                if image_name_path.name == item_path.name:
                    sample = item
                    base_sample_idx = i
                    break

        if sample is None:
            paths = [s["image_path"] for s in base_ds.samples]
            hint = paths[:10] if len(paths) > 10 else paths
            raise ValueError(
                f"No sample found for image name {image_name!r} (stem: {name_stem!r}). "
                f"Try index=..., or a relative path like one of: {hint}"
            )

        assert base_sample_idx is not None
        if patch_ds is not None:
            v = patch_variant
            if v < 0 or v >= patch_ds.variants_per_image:
                raise IndexError(
                    f"patch_variant {v} out of range for [0, {patch_ds.variants_per_image})"
                )
            flat_idx = base_sample_idx * patch_ds.variants_per_image + v
        else:
            flat_idx = base_sample_idx

    assert sample is not None and base_sample_idx is not None and flat_idx is not None

    patch_info = ""
    if patch_ds is not None:
        v_in_image = flat_idx % patch_ds.variants_per_image
        crop_v, flipped = patch_ds.decode_variant_index(v_in_image)
        patch_info = f" patch_var={v_in_image} crop_slot={crop_v} mirror={flipped}"

    print(
        f"visualization: flat_idx={flat_idx} base_idx={base_sample_idx}{patch_info} path={image_name_path} "
        f"precomputed density={use_precomputed_density} use_dataset_item={use_dataset_item} adaptive={adaptive} "
        f"beta={base_ds.density_beta} k={base_ds.density_k} min_sigma={base_ds.density_min_sigma}"
    )

    image_path = sample["image_path"]
    if base_ds.root is not None:
        image_path = base_ds.root / image_path
    else:
        image_path = Path(image_path)

    raw_image = _load_image(image_path)
    _, Hgt, Wgt = raw_image.shape
    shape = (Hgt, Wgt)
    ann_type = _parse_annotation_type(sample["annotation_type"])
    annotations = sample["annotations"]

    density_adaptive_label = (
        base_ds.density_geometry_adaptive if use_dataset_item else adaptive
    )

    if use_dataset_item:
        batch = dataset[flat_idx]
        masked_image = batch["image"]
        density = batch["density"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        count = float(batch["count"].item())
        infer_image = batch["image"]
    else:
        if use_precomputed_density:
            if base_ds.density_map_dir is None:
                raise ValueError(
                    "use_precomputed_density=True but dataset.density_map_dir is None. "
                    "Either set density_map_dir when creating the dataset or disable use_precomputed_density."
                )
            density_path = _density_map_path_for_sample(
                sample,
                base_ds.root,
                base_ds.density_map_dir,
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
                sigma=base_ds.density_sigma,
                geometry_adaptive=adaptive,
                beta=base_ds.density_beta,
                k=base_ds.density_k,
                min_sigma=base_ds.density_min_sigma,
                sigma_scale_bbox=base_ds.density_sigma_scale_bbox,
                sigma_from_seg_area=base_ds.density_sigma_from_seg_area,
                fixed_sigma_seg=base_ds.density_fixed_sigma_seg,
            )

        count = float(np.sum(density))

        inst_mask, _ = generate_instance_mask(
            shape,
            ann_type,
            annotations,
            dot_box_size=base_ds.mask_dot_box_size,
            dot_box_aspect=base_ds.mask_dot_box_aspect,
            dot_sigma_to_box=base_ds.mask_dot_sigma_to_box,
            dot_sigma=base_ds.mask_dot_sigma,
            dot_geometry_adaptive=base_ds.mask_dot_geometry_adaptive,
            dot_geometry_beta=base_ds.mask_dot_geometry_beta,
            dot_geometry_k=base_ds.mask_dot_geometry_k,
            dot_geometry_min_sigma=base_ds.mask_dot_geometry_min_sigma,
            dot_geometry_max_sigma=base_ds.mask_dot_geometry_max_sigma,
            mask_object_ratio=base_ds.mask_object_ratio,
            dot_mask_style=base_ds.mask_dot_style,
        )
        mask_t = torch.from_numpy(inst_mask.astype(np.float32)).unsqueeze(0)
        masked_image = raw_image * (1.0 - mask_t.clamp(0.0, 1.0))
        infer_image = raw_image

    # Optional model prediction
    pred_arr: Optional[np.ndarray] = None
    pred_count: Optional[float] = None
    if model is not None:
        model.eval()
        model = model.to(device)
        infer_b = infer_image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(infer_b)  # (1,1,H,W) or similar
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()
            if pred.ndim == 4 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            scale = float(pred_density_scale)
            pred_count = float(pred.sum().item() / scale)

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
                # print(f"gt density interpolated from {H_gt}x{W_gt} to {pred_h}x{pred_w}: {density.shape}")
                count = float(density.sum())
            # Same units as GT panel (raw density, integral ≈ count); model output is in training scale.
            pred_arr = (pred.squeeze().float() / scale).numpy()

    has_dots = ann_type == AnnotationType.DOT and annotations

    # Layout: row0 = Image | Masked image | [Dots] | GT density | [Pred]
    #         row1 = empty … | numeric stats under each density map
    ncols = 3 + (1 if has_dots else 0) + (1 if pred_arr is not None else 0)
    wscale = max(1.0, ncols / 2.0)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(figsize[0] * wscale, figsize[1] * 1.45),
        gridspec_kw={"height_ratios": [5.0, 1.25], "hspace": 0.28},
        squeeze=False,
    )
    row0 = axes[0]
    row1 = axes[1]
    ax_im = row0[0]
    ax_masked = row0[1]
    idx = 2
    if has_dots:
        ax_dots = row0[idx]
        idx += 1
    else:
        ax_dots = None  # type: ignore[assignment]
    ax_den = row0[idx]
    gt_col = idx
    idx += 1
    if pred_arr is not None:
        ax_pred = row0[idx]
        pred_col = idx
    else:
        ax_pred = None  # type: ignore[assignment]
        pred_col = None

    img_cpu = raw_image.permute(1, 2, 0).detach().cpu().numpy()
    ax_im.imshow(img_cpu)
    spatial_mismatch = raw_image.shape != masked_image.shape
    title_suffix = (
        f" (file {raw_image.shape[-2]}×{raw_image.shape[-1]})"
        if spatial_mismatch
        else ""
    )
    idx_title = (
        f"Image (flat idx {flat_idx}, base {base_sample_idx})"
        if patch_ds is not None
        else f"Image (idx {flat_idx})"
    )
    ax_im.set_title(f"{idx_title}{title_suffix}")
    ax_im.axis("off")

    _mr = base_ds.mask_object_ratio
    if _mr is None or float(_mr) <= 0:
        ratio = "mask off"
    else:
        ratio = f"mask ratio={_mr}"
    masked_vis = masked_image
    if use_dataset_item:
        t = masked_vis.detach()
        if t.min() < -0.01 or t.max() > 1.01:
            # timm ViT default_cfg often uses mean=std=0.5 → tensor = (x - 0.5) / 0.5
            masked_vis = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    ax_masked.imshow(masked_vis.permute(1, 2, 0).detach().cpu().numpy())
    _spatial_note = (
        f"\ninput {masked_image.shape[-2]}×{masked_image.shape[-1]}"
        if spatial_mismatch
        else ""
    )
    _masked_title = (
        f"Masked image{_spatial_note}\n({ratio}, {base_ds.mask_mode}"
        + (", same as __getitem__)" if use_dataset_item else ")")
    )
    _masked_title += f"\ncount ≈ {count:.1f}"
    ax_masked.set_title(_masked_title)
    ax_masked.axis("off")

    if has_dots:
        ax_dots.imshow(img_cpu)
        xs = [p[0] for p in annotations]
        ys = [p[1] for p in annotations]
        ax_dots.scatter(xs, ys, c="lime", s=12, edgecolors="darkgreen", linewidths=0.5, zorder=5)
        ax_dots.set_title(f"Dots ({len(annotations)})")
        ax_dots.axis("off")

    im = ax_den.imshow(density, cmap=density_cmap)
    ax_den.set_title(
        f"GT density"
        + (
            f"\n(count ≈ {count:.1f})"
            f"\nadaptive={density_adaptive_label} "
            f"beta={base_ds.density_beta} "
            f"\nk={base_ds.density_k} "
            f"min_sigma={base_ds.density_min_sigma} "
        )
    )
    ax_den.axis("off")
    plt.colorbar(im, ax=ax_den, fraction=0.046, pad=0.04)

    if pred_arr is not None and ax_pred is not None:
        im_pred = ax_pred.imshow(pred_arr, cmap=density_cmap)
        ax_pred.set_title(f"Pred density (count ≈ {pred_count:.1f})")
        ax_pred.axis("off")
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    for j in range(ncols):
        row1[j].axis("off")
        row1[j].set_facecolor("0.97")

    row1[gt_col].text(
        0.5,
        0.5,
        _density_stats_text(density),
        transform=row1[gt_col].transAxes,
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
    )
    row1[gt_col].set_title("GT numbers", fontsize=9, pad=4)

    if pred_arr is not None and pred_col is not None:
        row1[pred_col].text(
            0.5,
            0.5,
            _density_stats_text(pred_arr),
            transform=row1[pred_col].transAxes,
            ha="center",
            va="center",
            fontsize=8,
            family="monospace",
        )
        row1[pred_col].set_title("Pred numbers", fontsize=9, pad=4)

    plt.tight_layout()

    # Save to file when requested (useful on headless/SSH setups).
    if save_dir is not None:
        # Same base image → same stem; patch wrapper needs flat_idx so files do not overwrite.
        if patch_ds is not None:
            save_path = Path(save_dir) / f"{name_stem}_patch{flat_idx:05d}.png"
        else:
            save_path = Path(save_dir) / f"{name_stem}.png"
        plt.savefig(str(save_path), dpi=150)
        plt.close()
    else:
        plt.show()
        plt.close()