from pathlib import Path
import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
import kagglehub


from data import density
from data.dataset import (
    PatchAugmentedDataset,
    DENSITY_MAP_DIR_AUTO,
    precompute_density_maps,
    visualize_csrnet_patch_augmented_dataset,
    visualize_image_and_density,
)
from data.shanghaitech import load_shanghaitech_dataset
from data.fsc147 import visualize_fsc147_density, load_fsc147_dataset
from data.transforms import (
    random_crop_transform,
    resize_transform,
    horizontal_flip_transform,
    color_jitter_transform,
    normalize_imagenet_transform,
    timm_eval_dict_transform,
    timm_train_dict_transform,
    vit_normalize_only_transform,
    compose_transforms,
)
from model.unet import UNetDensity
from model.csrnet import CSRNet, load_vgg16_frontend
from model.vit_density import ViTDensity
from training.train import train, DEFAULT_DENSITY_SCALE


def _load_vit_init_weights(
    model: ViTDensity,
    ckpt_path: str | Path,
    mode: str = "full",
    device: torch.device | str = "cpu",
) -> None:
    """Load ViT initialization checkpoint with optional backbone-only mode."""
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ViT init checkpoint not found: {path}")

    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {path}. Expected a state_dict or "
            "a dict containing model_state_dict."
        )

    if mode == "full":
        model.load_state_dict(state, strict=True)
        print(f"Loaded full ViT checkpoint from {path}.")
        return
    if mode != "backbone":
        raise ValueError(f"Unknown --vit-init-load-mode {mode!r}; use 'full' or 'backbone'.")

    enc_prefix = "encoder."
    enc_state = {
        k[len(enc_prefix) :]: v
        for k, v in state.items()
        if k.startswith(enc_prefix)
    }
    if not enc_state:
        raise ValueError(
            f"No encoder weights found in {path}. Expected keys starting with '{enc_prefix}'."
        )
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
    if unexpected:
        warnings.warn(
            f"Unexpected encoder keys while loading backbone from {path}: {unexpected[:8]}",
            stacklevel=1,
        )
    if missing:
        warnings.warn(
            f"Missing encoder keys while loading backbone from {path}: {missing[:8]}",
            stacklevel=1,
        )
    print(f"Loaded ViT backbone only from {path}.")


class _Tee:
    """Write to both a terminal stream and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train density estimation models.")
    parser.add_argument(
        "--model",
        type=str,
        default="csrnet",
        choices=["unet", "csrnet", "vit"],
        help="Which density model to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    
    parser.add_argument(
        "--count-loss-weight",
        type=float,
        default=0,
        help="Weight for the count loss term.",
    )
    parser.add_argument(
        "--invisible-density-weight",
        type=float,
        default=1.0,
        help="Invisible-region weight inside split density loss. 1.0 reproduces previous full-map MSE.",
    )
    parser.add_argument(
        "--invisible-count-weight",
        type=float,
        default=1.0,
        help="Weight on invisible-region count error vs visible in split count loss "
        "(L1 or MSE per region, then weighted sum).",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="shng-A",
        help="Short identifier for the dataset (used in saved filenames).",
    )
    parser.add_argument(
        "--data-part",
        type=str,
        default="part_B",
        choices=["part_A", "part_B"],
        help="ShanghaiTech partition to train/evaluate on.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Fraction of object instances to mask per image (0..1), or None to disable masking.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="inpaint",
        choices=["robust", "inpaint"],
        help="'robust': mask both image and density (robustness augmentation). "
        "'inpaint': mask image only, keep full density target (hallucination task).",
    )
    parser.add_argument(
        "--mask-dot-style",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Dot instance mask: 'box' rectangles; 'gaussian' uses CSRNet-sigma disks on the image "
        "and (with --mask-mode robust) subtracts each masked head's GT Gaussian from density "
        "instead of multiplying by (1 - mask).",
    )
    parser.add_argument(
        "--deterministic-masks",
        action="store_true",
        help="Use deterministic per-sample masked-instance selection.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Base seed for deterministic masks. Effective seed is (mask_seed + sample_idx).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models",
        help="Directory where trained models and curves are saved.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after N epochs without val MAE improvement. None = disabled (run all epochs).",
    )
    parser.add_argument(
        "--density-scale",
        type=float,
        default=DEFAULT_DENSITY_SCALE,
        help="Multiplies density MSE (raw pred vs raw GT): loss += scale * MSE(pred,gt). "
        "Does not scale tensors inside MSE. Count = pred.sum(). CSRNet: typically 1. "
        "viz pred_density_scale=1 for current loss; legacy scaled-target training may differ.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the pretrained encoder (VGG for CSRNet, ViT for vit).",
    )
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help="ViT only: train a strict linear probe head (1x1 on patch features) "
        "on top of a frozen encoder.",
    )
    parser.add_argument(
        "--unfreeze-backbone-after-epoch",
        type=int,
        default=None,
        metavar="N",
        help="After epoch N, set ViT encoder.requires_grad=True and add it to the optimizer. "
        "Use with --freeze-encoder (e.g. N=5). Ignored for models without .encoder.",
    )
    parser.add_argument(
        "--vit-full-timm-aug",
        action="store_true",
        help="ViT train: use timm-style RandomResizedCrop+ColorJitter (stronger than default).",
    )
    parser.add_argument(
        "--vit-native-resolution",
        action="store_true",
        help="ViT only: train on fixed-pixel-size random crops (default 224x224) at the "
        "image's native resolution (no resize), and validate on full images at native "
        "resolution. Removes the train/val distribution shift that the default "
        "scale-fraction crop + 224x224 resize pipeline introduces. Forces val batch "
        "size = 1.",
    )
    parser.add_argument(
        "--vit-native-crop",
        type=int,
        default=224,
        help="Pixel size of the fixed square crop used by --vit-native-resolution. "
        "Must be a multiple of the ViT patch size (16 for vit_*_patch16_*).",
    )
    parser.add_argument(
        "--density-biased-crops",
        action="store_true",
        help="Sample training crop positions with probability proportional to the "
        "density mass inside the window, instead of uniformly. Requires a fixed-pixel "
        "crop (e.g. --vit-native-resolution). Avoids the 'mostly-empty crops' bias on "
        "concentrated-crowd datasets like ShanghaiTech part_A.",
    )
    parser.add_argument(
        "--density-bias-uniform-eps",
        type=float,
        default=0.1,
        help="Mixing weight on uniform fallback inside density-biased sampling "
        "(default 0.1 = 10%% of crops are still uniform so the model keeps seeing "
        "pure-background examples).",
    )
    parser.add_argument(
        "--density-bias-baseline",
        type=float,
        default=0.05,
        help="Additive uniform prior on the window-mass map, as a fraction of its max "
        "(default 0.05). Keeps low-density regions reachable by the multinomial.",
    )
    parser.add_argument(
        "--tiled-val",
        action="store_true",
        help="Validate with sliding-window tiled inference at native image resolution. "
        "Tiles are the same size as the training crop, so the ViT sees the exact "
        "input size (and attention length) it was trained on -- fixes the "
        "attention-length shift that --vit-native-resolution alone leaves in place "
        "for ShanghaiTech-scale full images. Requires --vit-native-resolution.",
    )
    parser.add_argument(
        "--tiled-val-tile",
        type=int,
        default=None,
        help="Tile size (pixels) for --tiled-val. Defaults to --vit-native-crop. "
        "Must be a multiple of the ViT patch size.",
    )
    parser.add_argument(
        "--tiled-val-overlap",
        type=int,
        default=48,
        help="Overlap (pixels) between adjacent tiles in --tiled-val (default 48). "
        "Should be >= ~3 sigma of the density-map Gaussians to avoid seam artifacts.",
    )
    parser.add_argument(
        "--tiled-val-max-batch",
        type=int,
        default=16,
        help="Max number of tiles forwarded at once per val image in --tiled-val. "
        "Lower this if you OOM on big images.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Default: 1e-7 for CSRNet, 1e-6 for UNet/ViT (override anytime).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.95,
        help="SGD momentum (ignored for Adam).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-3,
        help="Weight decay.",
    )
    parser.add_argument(
        "--single-image",
        action="store_true",
        help="Sanity-check mode: train/validate using a single image only.",
    )
    parser.add_argument(
        "--single-image-index",
        type=int,
        default=0,
        help="Index of image to use when --single-image is enabled.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation during training.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a *-latest.pth file saved during training. Restores model, optimizer, "
        "LR scheduler, epoch, best MAE, early-stopping counter, and history when present in the file.",
    )
    parser.add_argument(
        "--vit-init-checkpoint",
        type=str,
        default="trained_models/vit/vit-shng-part_A-0.3-inpaint-gaussian-random/vit-shng-0.3-inpaint-gaussian-random-best.pth",
        help="Optional checkpoint used to initialize ViT before training. "
        "Set to empty string to disable explicit init checkpoint loading.",
    )
    parser.add_argument(
        "--vit-init-load-mode",
        type=str,
        default="full",
        choices=["full", "backbone"],
        help="'full': load full ViTDensity state_dict. "
        "'backbone': load only encoder.* weights (useful with --linear-probe).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.lr is None:
        args.lr = 1e-7 if args.model == "csrnet" else 1e-6
    if args.model == "vit" and args.optimizer == "sgd":
        warnings.warn(
            "ViT fine-tuning with SGD often plateaus; prefer --optimizer adam.",
            stacklevel=1,
        )
    if (
        args.model == "vit"
        and args.freeze_encoder
        and args.unfreeze_backbone_after_epoch is None
    ):
        warnings.warn(
            "ViT encoder stays frozen for the whole run (no --unfreeze-backbone-after-epoch); "
            "only the decoder is trained.",
            stacklevel=1,
        )
    if args.linear_probe and args.model != "vit":
        raise ValueError("--linear-probe is only supported for --model vit.")
    if args.linear_probe and not (args.freeze_encoder==True and args.unfreeze_backbone_after_epoch is None):
        raise ValueError("--linear-probe requires --freeze-encoder and --unfreeze-backbone-after-epoch=None.")
    if args.model == "vit" and args.linear_probe and args.vit_init_load_mode == "full":
        # Linear probe changes decoder shape; full loading from a fine-tuned checkpoint
        # usually fails on decoder keys, so default to backbone-only loading.
        args.vit_init_load_mode = "backbone"
        print("Linear probe detected: switching --vit-init-load-mode to 'backbone'.")
    # Build log path: <output_dir>/<run_tag>/train-<date>.log
    mask_str = "nomsk" if args.mask_ratio is None else f"{args.mask_ratio}-{args.mask_mode or 'inpaint'}"
    if args.mask_ratio is not None and args.mask_dot_style != "box":
        mask_str = f"{mask_str}-{args.mask_dot_style}"
    mask_sampling_mode = "deterministic" if args.deterministic_masks else "random"
    if args.mask_ratio is not None:
        mask_str = f"{mask_str}-{mask_sampling_mode}"
    run_tag = f"{args.model}-{args.data_name}-{args.data_part}-{mask_str}"
    date_str = datetime.now().strftime("%Y-%m-%d-%Hh")

    log_dir = Path(args.output_dir) / args.model / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train-{date_str}.log"

    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"Logging to {log_path}")

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transform = compose_transforms(
    # #     # random_90deg_rotation_transform,
    #     # color_jitter_transform,
    #     normalize_imagenet_transform,
    # )


    # Choose which density model to train based on CLI flag.
    if args.model == "unet":
        model = UNetDensity()
    elif args.model == "csrnet":
        model = CSRNet()
        load_vgg16_frontend(model, freeze_frontend=args.freeze_encoder)
    elif args.model == "vit":
        model = ViTDensity(
            # encoder_name="vit_small_patch16_224.augreg_in21k_ft_in1k",
            encoder_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            freeze_encoder=args.freeze_encoder,
            output_activation="softplus",
            linear_probe=args.linear_probe,
        )
        if args.vit_init_checkpoint:
            _load_vit_init_weights(
                model,
                args.vit_init_checkpoint,
                mode=args.vit_init_load_mode,
                device=device,
            )
    else:
        raise ValueError(f"Unknown model {args.model!r}")


    train_transform = None
    eval_transform = None
    if args.tiled_val and args.model != "vit":
        raise ValueError("--tiled-val is only supported for --model vit.")
    if args.model == "vit":
        if args.vit_native_resolution:
            # Native-resolution recipe: train on fixed-pixel crops, validate on full images.
            # Train and val both see images at the same heads-per-pixel statistics.
            patch = model.patch_size
            if args.vit_native_crop % patch != 0:
                raise ValueError(
                    f"--vit-native-crop={args.vit_native_crop} must be a multiple of the ViT "
                    f"patch size ({patch})."
                )
            print(
                f"ViT native-resolution mode: train crops {args.vit_native_crop}x"
                f"{args.vit_native_crop} at native res, val at full image resolution."
            )
            train_transform = vit_normalize_only_transform(model.encoder)
            eval_transform = vit_normalize_only_transform(model.encoder)
            if args.tiled_val:
                tile = args.tiled_val_tile if args.tiled_val_tile is not None else args.vit_native_crop
                if tile % patch != 0:
                    raise ValueError(
                        f"--tiled-val-tile={tile} must be a multiple of the ViT patch "
                        f"size ({patch})."
                    )
                if args.tiled_val_overlap < 0 or args.tiled_val_overlap >= tile:
                    raise ValueError(
                        f"--tiled-val-overlap={args.tiled_val_overlap} must satisfy "
                        f"0 <= overlap < tile ({tile})."
                    )
                print(
                    f"Tiled validation: tile={tile}, overlap={args.tiled_val_overlap}, "
                    f"max_batch={args.tiled_val_max_batch}."
                )
        else:
            if args.tiled_val:
                raise ValueError(
                    "--tiled-val requires --vit-native-resolution so val images are fed "
                    "at native resolution (batch size 1, no resize/center-crop)."
                )
            # Default: timm resize+center-crop pipeline (ViT pretraining-style 224x224).
            # Note: this introduces a train/val distribution shift on dense crowd datasets
            # because the train side also crops fractional windows that get resized to 224.
            eval_transform = timm_eval_dict_transform(model.encoder)
            train_transform = timm_train_dict_transform(
                model.encoder,
                mode="full" if args.vit_full_timm_aug else "light",
            )

    dataset_kwargs = dict(
        root="./ShanghaiTech",
        part=[args.data_part],
        density_map_dir=DENSITY_MAP_DIR_AUTO,
        keep_original_image=False,
        density_geometry_adaptive=True,
        density_beta=0.3,
        density_k=3,
        density_min_sigma=4.0,
    )

    # CSRNet-style: train on full train_data, validate on test_data every epoch.
    train_full_dataset = load_shanghaitech_dataset(
        **dataset_kwargs,
        split="train_data",
        mask_object_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_dot_style=args.mask_dot_style,
        deterministic_masks=args.deterministic_masks,
        mask_seed=args.mask_seed,
        mask_dot_box_aspect=(3, 1),
        mask_dot_sigma_to_box=4.0,
        # transform=train_transform,
    )

    if args.single_image:
        # train_full_dataset.transform = train_transform
        if len(train_full_dataset) == 0:
            raise ValueError("Dataset is empty, cannot run single-image sanity check.")
        if args.single_image_index < 0 or args.single_image_index >= len(train_full_dataset):
            raise ValueError(
                f"--single-image-index must be in [0, {len(train_full_dataset) - 1}], "
                f"got {args.single_image_index}."
            )
        single_tfm = eval_transform if args.model == "vit" else train_transform
        train_dataset = Subset(
            load_shanghaitech_dataset(
                **dataset_kwargs,
                split="train_data",
                mask_object_ratio=None,
                transform=single_tfm,
            ),
            [args.single_image_index],
        )
        val_dataset = train_dataset
    else:
        # In native-resolution mode the cropping happens at native pixel size (no resize),
        # so heads/pixel is preserved. Otherwise we keep the legacy fractional-scale crops.
        if args.model == "vit" and args.vit_native_resolution:
            patch_kwargs = dict(
                random_crops_per_image=3,
                fixed_patch_scale=0.5,   # ignored when fixed_crop_size is set
                random_patch_scale=0.75,  # ignored when fixed_crop_size is set
                mirror=True,
                transform=train_transform,
                fixed_crop_size=(args.vit_native_crop, args.vit_native_crop),
                density_biased_crops=args.density_biased_crops,
                density_bias_uniform_eps=args.density_bias_uniform_eps,
                density_bias_baseline=args.density_bias_baseline,
            )
        else:
            if args.density_biased_crops:
                raise ValueError(
                    "--density-biased-crops requires a fixed-pixel crop. Pass "
                    "--vit-native-resolution (ViT) or remove --density-biased-crops."
                )
            patch_kwargs = dict(
                random_crops_per_image=3,
                fixed_patch_scale=0.5,
                random_patch_scale=0.75,
                mirror=True,
                transform=train_transform,
            )
        train_dataset = PatchAugmentedDataset(train_full_dataset, **patch_kwargs)
        val_dataset = load_shanghaitech_dataset(
            **dataset_kwargs,
            split="test_data",
            mask_object_ratio=None,
            # mask_object_ratio=args.mask_ratio,
            # mask_mode=args.mask_mode,
            # mask_dot_style=args.mask_dot_style,
            transform=eval_transform,
        )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    

    train_kw = dict(
        epochs=args.epochs,
        count_loss_weight=args.count_loss_weight,
        invisible_density_weight=args.invisible_density_weight,
        invisible_count_weight=args.invisible_count_weight,
        model_name=args.model,
        data_name=args.data_name,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_dot_style=args.mask_dot_style,
        mask_sampling_mode=mask_sampling_mode,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping_patience,
        density_scale=args.density_scale,
        gt_downsample="bilinear",
        batch_size=args.batch_size,
        validate_during_training=not args.no_validation,
        optimizer_type=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        unfreeze_backbone_after_epoch=args.unfreeze_backbone_after_epoch,
        resume_checkpoint=args.resume_checkpoint,
        log_dir=log_dir,
    )
    if args.model == "csrnet":
        train_kw["gt_downsample"] = "csrnet_cubic"
        train_kw["density_scale"] = 1.0

    if args.model == "vit" and args.vit_native_resolution:
        # Each val image keeps its native shape, so they cannot be batched together.
        train_kw["val_batch_size"] = 1

    if args.tiled_val:
        train_kw["tiled_val"] = True
        train_kw["tiled_val_tile"] = (
            args.tiled_val_tile if args.tiled_val_tile is not None else args.vit_native_crop
        )
        train_kw["tiled_val_overlap"] = args.tiled_val_overlap
        train_kw["tiled_val_max_batch"] = args.tiled_val_max_batch

    train(model, train_dataset, val_dataset, **train_kw)

   
    