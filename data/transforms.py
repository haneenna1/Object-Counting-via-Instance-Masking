from __future__ import annotations

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def resize_transform(sample, size=(512, 512)):
    image = sample["image"]
    _, h, w = image.shape
    new_h, new_w = size
    density = sample["density"]
    mask = sample["mask"]

    image = F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    density = F.interpolate(
        density.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    scale = (new_h * new_w) / (h * w)
    density = density / scale

    mask = F.interpolate(
        mask.unsqueeze(0),
        size=size,
        mode="nearest",
    ).squeeze(0)

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return sample


def random_crop_transform(sample, crop_size=(256, 256)):
    image = sample["image"]
    density = sample["density"]
    mask = sample["mask"]

    _, H, W = image.shape
    crop_h, crop_w = crop_size

    # ---- pad if needed ----
    pad_h = max(crop_h - H, 0)
    pad_w = max(crop_w - W, 0)

    b_im_sh = image.shape
    b_d_sh = density.shape
    b_msk_sh = mask.shape

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h))
        density = F.pad(density, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))

        H = H + pad_h
        W = W + pad_w

    # ---- random crop ----
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    image = image[:, top : top + crop_h, left : left + crop_w]
    density = density[:, top : top + crop_h, left : left + crop_w]
    mask = mask[:, top : top + crop_h, left : left + crop_w]

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return sample


def horizontal_flip_transform(sample, p: float = 0.5):
    """
    Randomly flip image / density / mask horizontally with probability p.
    """
    if random.random() < p:
        sample["image"] = torch.flip(sample["image"], dims=[2])
        sample["density"] = torch.flip(sample["density"], dims=[2])
        sample["mask"] = torch.flip(sample["mask"], dims=[2])
    return sample


def random_90deg_rotation_transform(sample):
    """
    Randomly rotate image / density / mask by 0, 90, 180, or 270 degrees.
    """
    k = random.randint(0, 3)
    if k == 0:
        return sample
    sample["image"] = torch.rot90(sample["image"], k, dims=[1, 2])
    sample["density"] = torch.rot90(sample["density"], k, dims=[1, 2])
    sample["mask"] = torch.rot90(sample["mask"], k, dims=[1, 2])
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

