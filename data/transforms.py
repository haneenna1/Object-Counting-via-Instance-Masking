from __future__ import annotations

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import InterpolationMode as TVInterpolationMode

# ---------------------------------------------------------------------------
# Interpolation policy (RGB vs density)
#
# timm ``create_transform`` / ImageNet ViT recipes often set ``interpolation='bicubic'``
# for **RGB** — keep that on the image branch (PIL) so inputs match pretraining.
#
# **Density maps** must not use bicubic: overshoot / ringing can produce negative values
# and break count preservation. Use only ``nearest`` or ``bilinear`` (linear in 2D).
# We use bilinear + explicit area correction after resizes; masks use nearest.
# ---------------------------------------------------------------------------
DENSITY_INTERPOLATE_MODE = "bilinear"
DENSITY_RESIZED_CROP_INTERP = TVInterpolationMode.BILINEAR
MASK_INTERPOLATE_MODE = "nearest"
MASK_RESIZED_CROP_INTERP = TVInterpolationMode.NEAREST


def _update_sample_count(sample):
    if "count" in sample:
        sample["count"] = torch.tensor(
            float(sample["density"].sum().item()),
            dtype=sample["density"].dtype,
            device=sample["density"].device,
        )
    return sample


def crop_sample(sample, top: int, left: int, crop_h: int, crop_w: int):
    image = sample["image"]
    density = sample["density"]
    mask = sample["mask"]

    sample["image"] = image[:, top : top + crop_h, left : left + crop_w]
    sample["density"] = density[:, top : top + crop_h, left : left + crop_w]
    sample["mask"] = mask[:, top : top + crop_h, left : left + crop_w]

    if "original_image" in sample:
        original_image = sample["original_image"]
        sample["original_image"] = original_image[
            :, top : top + crop_h, left : left + crop_w
        ]

    return _update_sample_count(sample)


def resize_transform(sample, size=(512, 512), image_mode: str = "bilinear"):
    image = sample["image"]
    _, h, w = image.shape
    new_h, new_w = size
    density = sample["density"]
    mask = sample["mask"]

    if image_mode in ("bilinear", "bicubic"):
        image = F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode=image_mode,
            align_corners=False,
        ).squeeze(0)
    else:
        image = F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode=image_mode,
        ).squeeze(0)

    density = F.interpolate(
        density.unsqueeze(0),
        size=size,
        mode=DENSITY_INTERPOLATE_MODE,
        align_corners=False,
    ).squeeze(0)
    scale = (new_h * new_w) / (h * w)
    density = density / scale

    mask = F.interpolate(
        mask.unsqueeze(0),
        size=size,
        mode=MASK_INTERPOLATE_MODE,
    ).squeeze(0)

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return _update_sample_count(sample)


def random_crop_transform(sample, crop_size=(256, 256)):
    image = sample["image"]
    density = sample["density"]
    mask = sample["mask"]

    _, H, W = image.shape
    crop_h, crop_w = crop_size

    # ---- pad if needed ----
    pad_h = max(crop_h - H, 0)
    pad_w = max(crop_w - W, 0)

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h))
        density = F.pad(density, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))

        H = H + pad_h
        W = W + pad_w

    # ---- random crop ----
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return crop_sample(sample, top=top, left=left, crop_h=crop_h, crop_w=crop_w)


def horizontal_flip_transform(sample, p: float = 0.5):
    """
    Randomly flip image / density / mask horizontally with probability p.
    """
    if random.random() < p:
        sample["image"] = torch.flip(sample["image"], dims=[2])
        sample["density"] = torch.flip(sample["density"], dims=[2])
        sample["mask"] = torch.flip(sample["mask"], dims=[2])
    return sample


def color_jitter_transform(
    sample,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.02,
):
    """
    Apply simple color jitter to the image only (not density or mask).
    """
    img = sample["image"]
    # TF expects (C,H,W) in [0,1]; our image already satisfies that.
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = TF.adjust_brightness(img, factor)
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        img = TF.adjust_contrast(img, factor)
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        img = TF.adjust_saturation(img, factor)
    if hue > 0:
        delta = random.uniform(-hue, hue)
        img = TF.adjust_hue(img, delta)
    sample["image"] = img.clamp(0.0, 1.0)
    return sample


def compose_transforms(*transforms):
    """
    Compose multiple sample-level transforms into one.
    """
    def _transform(sample):
        for t in transforms:
            if t is not None:
                sample = t(sample)
        return sample

    return _transform


def normalize_imagenet_transform(
    sample,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Apply ImageNet mean/std normalization to sample["image"] only.
    Density maps and masks are left untouched.

    Assumes sample["image"] is a float tensor in [0, 1] with shape (3, H, W).
    """
    img = sample["image"]
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(3, 1, 1)
    sample["image"] = (img - mean_t) / std_t
    return sample


def normalize_transform(
    sample,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    """
    Apply channel-wise normalization to sample["image"] only.
    """
    img = sample["image"]
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(3, 1, 1)
    sample["image"] = (img - mean_t) / std_t
    return sample


# timm `resolve_model_data_config` for vit_*_patch16_224.augreg_in21k_ft_in1k (and similar augreg FT weights).
VIT_AUGREG_IN21K_FT_IN1K_DATA_CONFIG = {
    "input_size": (3, 224, 224),
    "interpolation": "bicubic",
    "mean": (0.5, 0.5, 0.5),
    "std": (0.5, 0.5, 0.5),
    "crop_pct": 0.9,
    "crop_mode": "center",
}


def resize_shortest_edge_transform(sample, size: int, image_mode: str = "bicubic"):
    """
    Match torchvision/timm eval resize: shortest spatial edge becomes `size`, aspect ratio preserved.
    """
    image = sample["image"]
    _, h, w = image.shape
    short = min(h, w)
    scale = size / short
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return resize_transform(sample, size=(new_h, new_w), image_mode=image_mode)


def trim_border_transform(sample, border_px: int):
    """
    Remove a uniform border from image, density, and mask (matches timm ``TrimBorder``).
    """
    if border_px <= 0:
        return sample
    _, h, w = sample["image"].shape
    b = border_px
    inner_h, inner_w = h - 2 * b, w - 2 * b
    if inner_h < 1 or inner_w < 1:
        return sample
    return crop_sample(sample, top=b, left=b, crop_h=inner_h, crop_w=inner_w)


def resize_to_fit_inside_box_transform(
    sample,
    max_h: int,
    max_w: int,
    image_mode: str = "bilinear",
):
    """
    Resize keeping aspect ratio so H <= max_h and W <= max_w (timm ``ResizeKeepRatio``-style).
    Density integral preserved via ``resize_transform``.
    """
    _, h, w = sample["image"].shape
    if h <= 0 or w <= 0:
        return sample
    scale = min(max_h / h, max_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return resize_transform(sample, size=(new_h, new_w), image_mode=image_mode)


def center_crop_transform(sample, crop_h: int, crop_w: int):
    """
    Center crop to (crop_h, crop_w). Pads with zeros if the current map is smaller (timm/torchvision behavior).
    """
    _, h, w = sample["image"].shape
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad = (pad_left, pad_right, pad_top, pad_bottom)
        sample["image"] = F.pad(sample["image"], pad, mode="constant", value=0.0)
        sample["density"] = F.pad(sample["density"], pad, mode="constant", value=0.0)
        sample["mask"] = F.pad(sample["mask"], pad, mode="constant", value=0.0)
        if "original_image" in sample:
            sample["original_image"] = F.pad(
                sample["original_image"], pad, mode="constant", value=0.0
            )
        h, w = sample["image"].shape[1], sample["image"].shape[2]
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return crop_sample(sample, top=top, left=left, crop_h=crop_h, crop_w=crop_w)


def random_resized_crop_transform(
    sample,
    size=(224, 224),
    scale=(0.08, 1.0),
    ratio=(0.75, 4.0 / 3.0),
    image_mode: str = "bicubic",
):
    """
    Same geometry as timm `RandomResizedCropAndInterpolation` defaults for ImageNet training.
    """
    img = sample["image"]
    top, left, h, w = RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
    sample = crop_sample(sample, top=top, left=left, crop_h=h, crop_w=w)
    return resize_transform(sample, size=size, image_mode=image_mode)

def _chw_float01_to_pil(img: torch.Tensor):
    """(3,H,W) float in [0,1] → PIL RGB."""
    return TF.to_pil_image(img.clamp(0.0, 1.0))


def _resize_density_mask_for_tv_resize(
    density: torch.Tensor,
    mask: torch.Tensor,
    op,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror ``torchvision.transforms.Resize`` for (1,H,W) maps; preserve count integral.

    Image resize may follow the op's bicubic setting in ``create_transform``; density here
    always uses ``DENSITY_INTERPOLATE_MODE`` (bilinear), never bicubic.
    """
    _, h, w = density.shape
    size = op.size
    if isinstance(size, int):
        short = min(h, w)
        scale = size / short
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
    elif isinstance(size, (list, tuple)):
        if len(size) == 1:
            s = int(size[0])
            short = min(h, w)
            scale = s / short
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
        elif len(size) == 2:
            new_h, new_w = int(size[0]), int(size[1])
        else:
            raise NotImplementedError(f"Resize.size {size!r} not supported for density/mask pairing.")
    else:
        raise NotImplementedError(f"Resize.size {size!r} not supported for density/mask pairing.")
    if getattr(op, "max_size", None) is not None:
        raise NotImplementedError("Resize.max_size for density/mask pairing is not implemented.")
    area_old = float(h * w)
    area_new = float(new_h * new_w)
    density = F.interpolate(
        density.unsqueeze(0),
        size=(new_h, new_w),
        mode=DENSITY_INTERPOLATE_MODE,
        align_corners=False,
    ).squeeze(0)
    density = density * (area_old / area_new)
    mask = F.interpolate(
        mask.unsqueeze(0),
        size=(new_h, new_w),
        mode=MASK_INTERPOLATE_MODE,
    ).squeeze(0)
    return density, mask


def _center_crop_density_mask_for_tv_center_crop(
    density: torch.Tensor,
    mask: torch.Tensor,
    op,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``torchvision.transforms.CenterCrop`` (with zero-pad if smaller, like torchvision)."""
    size = op.size
    if isinstance(size, int):
        ch = cw = int(size)
    else:
        ch, cw = int(size[0]), int(size[1])
    _, h, w = density.shape
    pad_h = max(ch - h, 0)
    pad_w = max(cw - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad = (pad_left, pad_right, pad_top, pad_bottom)
        density = F.pad(density, pad, mode="constant", value=0.0)
        mask = F.pad(mask, pad, mode="constant", value=0.0)
        h, w = density.shape[1], density.shape[2]
    top = (h - ch) // 2
    left = (w - cw) // 2
    density = density[:, top : top + ch, left : left + cw]
    mask = mask[:, top : top + ch, left : left + cw]
    return density, mask


def _apply_trim_border_density_mask(
    density: torch.Tensor,
    mask: torch.Tensor,
    op,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same crop box as ``timm.data.transforms.TrimBorder.forward`` (matches PIL image)."""
    bs = int(op.border_size)
    _, h, w = density.shape
    top = left = bs
    top = min(top, h)
    left = min(left, h)  # timm uses h here as well; keep parity with timm
    height = max(0, h - 2 * bs)
    width = max(0, w - 2 * bs)
    density = density[:, top : top + height, left : left + width]
    mask = mask[:, top : top + height, left : left + width]
    return density, mask


def _spatial_ops_before_maybe_to_tensor(compose_tf) -> list:
    ops = []
    for t in compose_tf.transforms:
        if type(t).__name__ == "MaybeToTensor":
            break
        ops.append(t)
    return ops


def _apply_timm_spatial_ops_to_density_and_mask(
    density: torch.Tensor,
    mask: torch.Tensor,
    spatial_ops: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    d, m = density, mask
    for op in spatial_ops:
        name = type(op).__name__
        if name == "TrimBorder":
            d, m = _apply_trim_border_density_mask(d, m, op)
        elif name == "Resize":
            d, m = _resize_density_mask_for_tv_resize(d, m, op)
        elif name == "CenterCrop":
            d, m = _center_crop_density_mask_for_tv_center_crop(d, m, op)
        else:
            raise NotImplementedError(
                f"Spatial transform {name!r} in timm create_transform is not implemented for "
                f"paired density/mask. Extend _apply_timm_spatial_ops_to_density_and_mask."
            )
    return d, m


def _rrc_interpolation_to_tv(rrc) -> TVInterpolationMode:
    """
    Timm RRC → torchvision interpolation for **RGB only**.

    Density uses ``DENSITY_RESIZED_CROP_INTERP`` (bilinear), never bicubic.
    """
    interp = rrc.interpolation
    if isinstance(interp, (tuple, list)):
        raise NotImplementedError(
            "Random interpolation in RandomResizedCropAndInterpolation is not supported when "
            "pairing density on tensors; use a single interpolation mode in timm config."
        )
    from timm.data.transforms import interp_mode_to_str

    s = interp_mode_to_str(interp).lower()
    mapping = {
        "nearest": TVInterpolationMode.NEAREST,
        "bilinear": TVInterpolationMode.BILINEAR,
        "bicubic": TVInterpolationMode.BICUBIC,
    }
    if s not in mapping:
        raise NotImplementedError(f"Unsupported RRC interpolation {s!r}")
    return mapping[s]

def vit_normalize_only_transform(timm_model):
    """
    Normalize ``sample['image']`` with the encoder's expected mean/std and **do nothing
    spatial** to image, density, or mask.

    Use this with native-resolution training/validation: the model receives the image at
    its true resolution (no resize, no center-crop), so heads/pixel and heads/ViT-token
    are preserved end-to-end. Pair with ``ViTDensity(dynamic_img_size=True)`` (default) on
    the model side and a fixed-size random-crop dataset on the train side.

    Validation with this transform requires ``batch_size=1`` because images have
    different native sizes.
    """
    import timm.data as tdata

    cfg = tdata.resolve_model_data_config(timm_model)
    mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=torch.float32).view(3, 1, 1)

    def _transform(sample):
        s = dict(sample)
        img = s["image"]
        m = mean.to(device=img.device, dtype=img.dtype)
        sd = std.to(device=img.device, dtype=img.dtype)
        s["image"] = (img - m) / sd
        return s

    return _transform


def timm_eval_dict_transform(timm_model):
    """
    Sample dict transform using ``timm.data.create_transform(..., is_training=False)`` for
    ``image`` (PIL path; uses cfg interpolation, often **bicubic**, matching ViT pretraining).

    The same *spatial* geometry (trim / resize / center crop) is replayed on ``density``
    with **bilinear** resampling only (never bicubic — avoids ringing/negative mass) and
    multiply by ``old_area / new_area`` after each resize. ``mask`` uses **nearest**.
    ``ToTensor`` + ``Normalize`` apply only to the image via timm.
    """
    import inspect

    import timm.data as tdata

    cfg = tdata.resolve_model_data_config(timm_model)
    if cfg.get("crop_mode") == "border":
        raise NotImplementedError(
            "timm crop_mode='border' is not supported for dict samples with density; "
            "use crop_mode 'center' or 'squash' (or None)."
        )
    allowed = set(inspect.signature(tdata.create_transform).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    img_tf = tdata.create_transform(**kwargs, is_training=False)
    spatial_ops = _spatial_ops_before_maybe_to_tensor(img_tf)

    def _transform(sample):
        s = dict(sample)
        image = s["image"]
        dens2, mask2 = _apply_timm_spatial_ops_to_density_and_mask(
            s["density"], s["mask"], spatial_ops
        )
        s["image"] = img_tf(_chw_float01_to_pil(image))
        s["density"] = dens2
        s["mask"] = mask2
        return _update_sample_count(s)

    return _transform


def timm_train_dict_transform(timm_model, *, mode: str = "light"):
    """
    Training transforms from ``resolve_model_data_config`` + ``create_transform``.

    - ``light`` / ``none``: same pipeline as eval — ``create_transform(..., is_training=False)``
      on the image, mirrored spatial ops on density/mask.
    - ``full``: ``create_transform(..., is_training=True, separate=True)`` — spatial
      (``RandomResizedCropAndInterpolation`` + ``RandomHorizontalFlip``) applied on tensors
      with the same crop/resize params for image, density, and mask. RGB ``resized_crop``
      uses timm's interpolation (e.g. bicubic); density uses ``DENSITY_RESIZED_CROP_INTERP``.
      Density is scaled by ``crop_pixel_area / output_pixel_area`` after the resized crop.
      Color jitter + ``MaybeToTensor`` + ``Normalize`` run on the image via timm (PIL).
    """
    import inspect

    import timm.data as tdata
    from timm.data.transforms import RandomResizedCropAndInterpolation
    from torchvision.transforms import RandomHorizontalFlip

    if mode in ("light", "none"):
        return timm_eval_dict_transform(timm_model)

    if mode != "full":
        raise ValueError(f"mode must be 'light', 'full', or 'none', got {mode!r}")

    cfg = tdata.resolve_model_data_config(timm_model)
    if cfg.get("crop_mode") == "border":
        raise NotImplementedError(
            "timm crop_mode='border' is not supported for dict samples with density; "
            "use crop_mode 'center' or 'squash' (or None)."
        )
    allowed = set(inspect.signature(tdata.create_transform).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    spatial_c, color_c, final_c = tdata.create_transform(
        **kwargs, is_training=True, separate=True
    )
    if len(spatial_c.transforms) != 2:
        raise NotImplementedError(
            "Unexpected timm train spatial compose; expected RandomResizedCropAndInterpolation "
            "then RandomHorizontalFlip."
        )
    rrc = spatial_c.transforms[0]
    hflip = spatial_c.transforms[1]
    if not isinstance(rrc, RandomResizedCropAndInterpolation):
        raise NotImplementedError(
            f"Expected RandomResizedCropAndInterpolation, got {type(rrc).__name__}."
        )
    if not isinstance(hflip, RandomHorizontalFlip):
        raise NotImplementedError(
            f"Expected RandomHorizontalFlip, got {type(hflip).__name__}."
        )
    image_interp = _rrc_interpolation_to_tv(rrc)
    out_h, out_w = int(rrc.size[0]), int(rrc.size[1])
    hflip_p = float(getattr(hflip, "p", 0.0))

    def _full(sample):
        s = dict(sample)
        img = s["image"]
        dens = s["density"]
        mask = s["mask"]
        top, left, h0, w0 = rrc.get_params(img, rrc.scale, rrc.ratio)
        img_t = TF.resized_crop(img, top, left, h0, w0, [out_h, out_w], image_interp)
        dens_t = TF.resized_crop(
            dens, top, left, h0, w0, [out_h, out_w], DENSITY_RESIZED_CROP_INTERP
        )
        mask_t = TF.resized_crop(
            mask, top, left, h0, w0, [out_h, out_w], MASK_RESIZED_CROP_INTERP
        )
        crop_area = float(h0 * w0)
        out_area = float(out_h * out_w)
        dens_t = dens_t * (crop_area / out_area)
        if hflip_p > 0.0 and torch.rand(1).item() < hflip_p:
            img_t = torch.flip(img_t, dims=[2])
            dens_t = torch.flip(dens_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[2])
        pil_img = _chw_float01_to_pil(img_t)
        pil_img = color_c(pil_img)
        s["image"] = final_c(pil_img)
        s["density"] = dens_t
        s["mask"] = mask_t
        return _update_sample_count(s)

    return _full
