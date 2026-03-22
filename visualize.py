

from data.dataset import visualize_image_and_density
from data.shanghaitech import load_shanghaitech_dataset
from model.csrnet import CSRNet
import torch

# # visualize_csrnet_patch_augmented_dataset(
# #     csrnet_patch_dataset,
# #     base_idx=0,
# #     include_mirrored=True,
# #     save_path="csrnet_patch_visualization.png",
# #     show=True,
# # )
if __name__ == "__main__":
    model = CSRNet()
    model.load_state_dict(torch.load("trained_models/csrnet-shng-nomsk.pth"))

    dataset = load_shanghaitech_dataset(
        root="/home/haneenn/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech",
        part=["part_A"],
        split="test_data",
    )
    # visualize_csrnet_patch_augmented_dataset(
    #     csrnet_patch_dataset,
    #     base_idx=0,
    #     include_mirrored=True,
    #     save_path="csrnet_patch_visualization.png",
    #     show=True,
    # )
    for i in range(1,len(dataset)):
        visualize_image_and_density(
            dataset,
            f"part_A/test_data/density_maps/IMG_{i}.jpg",
            use_precomputed_density=False,
            adaptive=True,
            pred_density_scale=1,
            save_path=f"vis/csr-shng-nomsk/IMG_{i}_tst.png",
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