from __future__ import annotations

import random

import torch
import torch.nn.functional as F


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


def random_crop_transform(sample, crop_size=(512, 512)):
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

    if (
        image.shape != (3, 512, 512)
        or mask.shape != (1, 512, 512)
        or density.shape != (1, 512, 512)
    ):
        print("before image shape:", b_im_sh, "after image shape:", image.shape)
        print("before density shape:", b_d_sh, "after density shape:", density.shape)
        print("before mask shape:", b_msk_sh, "after mask shape:", mask.shape)

    sample["image"] = image
    sample["density"] = density
    sample["mask"] = mask

    return sample

