

from data.dataset import PatchAugmentedDataset, visualize_csrnet_patch_augmented_dataset, visualize_image_and_density
from data.shanghaitech import load_shanghaitech_dataset
from model.csrnet import CSRNet
import torch
from data.transforms import compose_transforms, normalize_imagenet_transform

# # visualize_csrnet_patch_augmented_dataset(
# #     csrnet_patch_dataset,
# #     base_idx=0,
# #     include_mirrored=True,
# #     save_path="csrnet_patch_visualization.png",
# #     show=True,
# # )
if __name__ == "__main__":
    model = CSRNet()
    model.load_state_dict(torch.load("trained_models/csrnet/csrnet-shng-nomsk-best.pth"))

    transform = compose_transforms(
        normalize_imagenet_transform,
    )

    dataset = load_shanghaitech_dataset(
        root="/home/haneenn/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech",
        part=["part_B"],
        split="test_data",
        density_geometry_adaptive=True,
        density_beta=0.3,
        density_k=3,
        density_min_sigma=4.0,
        transform=transform,
        # mask_object_ratio=0.5,
        # mask_dot_box_aspect=(2, 1),
    )
    for i in range(1, len(dataset)):
        visualize_image_and_density(
            dataset,
            index=i,
            use_precomputed_density=True,
            # adaptive=True,
            pred_density_scale=1,
            save_dir=f"vis/csr-shngA-nomsk/part_B_test",
            model=model,
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