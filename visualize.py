


from data.dataset import PatchAugmentedDataset, visualize_csrnet_patch_augmented_dataset, visualize_image_and_density
from data.shanghaitech import load_shanghaitech_dataset
from model.csrnet import CSRNet
from model.vit_density import ViTDensity
import torch
from data.transforms import compose_transforms, normalize_imagenet_transform, timm_eval_dict_transform, timm_train_dict_transform, vit_normalize_only_transform
import os
import argparse
# # visualize_csrnet_patch_augmented_dataset(
# #     csrnet_patch_dataset,
# #     base_idx=0,
# #     include_mirrored=True,
# #     save_path="csrnet_patch_visualization.png",
# #     show=True,
# # )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="part_A", choices=["part_A", "part_B"])
    parser.add_argument("--split", type=str, default="train_data", choices=["train_data", "test_data"])
    parser.add_argument("--mask-ratio", type=float, default=0)
    parser.add_argument("--save-dir", type=str, default="vis/shng-A-30-inpaint-gaussian/part_A_train")
    parser.add_argument("--pred-density-scale", type=float, default=100)
    parser.add_argument("--use-precomputed-density", type=bool, default=True)
    parser.add_argument("--use-dataset-item", type=bool, default=True)
    parser.add_argument("--model", type=str, default="csrnet", choices=["csrnet", "vit"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # model = CSRNet()
    # model.load_state_dict(torch.load("trained_models/csrnet/csrnet-shng-nomsk-best.pth"))
    model = ViTDensity(
        encoder_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True,
        output_activation="softplus"
    )
    # eval_transform = timm_eval_dict_transform(model.encoder)
    # train_transform = timm_train_dict_transform(model.encoder)
    # model.load_state_dict(torch.load("trained_models/vit/vit-shng-part_A-0.3-inpaint-gaussian-random/vit-shng-0.3-inpaint-gaussian-random-best.pth"))
    eval_transform = vit_normalize_only_transform(model.encoder)


    dataset = load_shanghaitech_dataset(
        root="./ShanghaiTech",
        part=[args.part],
        split=args.split,
        density_geometry_adaptive=True,
        density_beta=0.3,
        density_k=3,
        density_min_sigma=4.0, 
        mask_object_ratio=args.mask_ratio,
        mask_mode="inpaint",
        mask_dot_style="gaussian",
        transform=eval_transform,
    )
    patch_dataset = PatchAugmentedDataset(
        dataset,
        random_crops_per_image=3,
        fixed_crop_size=(224, 224),
        fixed_patch_scale=0.5,   # corners: quarter-area style
        random_patch_scale=0.75,  # randoms: larger windows
        mirror=True,
        density_biased_crops=True,
        transform=eval_transform,
    )

    # visualize_csrnet_patch_augmented_dataset(
    #     patch_dataset,
    #     base_idx=0,
    #     include_mirrored=True,
    #     save_path="new_csrnet_patch_visualization.png",
    #     show=True,
    # )
    # exit()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(patch_dataset)):
        visualize_image_and_density(
            patch_dataset,
            index=i,
            use_precomputed_density=args.use_precomputed_density,
            pred_density_scale=args.pred_density_scale,
            save_dir=save_dir,
            use_dataset_item=args.use_dataset_item,
            # model=model,
        )
    # visualize_image_and_density(
    #     dataset,
    #     "part_A/test_data/density_maps/IMG_1.jpg",
    #     use_precomputed_density=False,
    #     adaptive=True,
    #     pred_density_scale=1,
    #     save_path="visualization_adaptive.png",
    #     model=model,
    # ) 