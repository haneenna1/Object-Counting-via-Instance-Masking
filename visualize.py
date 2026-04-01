

from data.dataset import PatchAugmentedDataset, visualize_csrnet_patch_augmented_dataset, visualize_image_and_density
from data.shanghaitech import load_shanghaitech_dataset
from model.csrnet import CSRNet
from model.vit_density import ViTDensity
import torch
from data.transforms import compose_transforms, normalize_imagenet_transform, timm_eval_dict_transform, timm_train_dict_transform
import os
# # visualize_csrnet_patch_augmented_dataset(
# #     csrnet_patch_dataset,
# #     base_idx=0,
# #     include_mirrored=True,
# #     save_path="csrnet_patch_visualization.png",
# #     show=True,
# # )
if __name__ == "__main__":
    # model = CSRNet()
    # model.load_state_dict(torch.load("trained_models/csrnet/csrnet-shng-nomsk-best.pth"))
    model = ViTDensity(
        encoder_name="vit_small_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True,
    )
    eval_transform = timm_eval_dict_transform(model.encoder)
    train_transform = timm_train_dict_transform(model.encoder, mode="full")
    # model.load_state_dict(torch.load("trained_models/vit/vit-shng-B-nomsk/vit-shng-B-nomsk-best.pth"))


    dataset = load_shanghaitech_dataset(
        root="/home/haneenn/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech",
        part=["part_B"],
        split="train_data",
        density_geometry_adaptive=True,
        density_beta=0.3,
        density_k=3,
        density_min_sigma=4.0,
        transform=eval_transform,
        mask_object_ratio=0.5,
        mask_dot_box_aspect=(2, 1),
        mask_mode="robust",
    )

    patch_dataset = PatchAugmentedDataset(
        dataset,
        random_crops_per_image=5,
        mirror=True,
    )

    visualize_csrnet_patch_augmented_dataset(
        patch_dataset,
        base_idx=0,
        include_mirrored=True,
        save_path="new_csrnet_patch_visualization.png",
        show=True,
    )

    # save_dir = f"vis/shng-B-50msk/part_B_test"
    # os.makedirs(save_dir, exist_ok=True)
    # for i in range(1, len(dataset)):
    #     visualize_image_and_density(
    #         patch_dataset,
    #         index=i,
    #         # image_name="part_B/test_data/IMG_1.jpg",
    #         use_precomputed_density=True,
    #         # adaptive=True,
    #         # pred_density_scale=10,
    #         save_dir=save_dir,
    #         # model=model,
    #     )
    # visualize_image_and_density(
    #     dataset,
    #     "part_A/test_data/density_maps/IMG_1.jpg",
    #     use_precomputed_density=False,
    #     adaptive=True,
    #     pred_density_scale=1,
    #     save_path="visualization_adaptive.png",
    #     model=model,
    # ) 