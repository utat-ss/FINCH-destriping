"""img_gen.py.

Contains functions to apply image patch extraction technique to build more imaged from existing ones; use to increase size of dataset

Author(s): Dav Vrat Chadha

Usage: new_images = img_gen(image, patch_size: int = 32, height: int = 620, width: int = 620, spectral_depth: int = 3, num: int = 1) -> list
Output: list of images that can be saved as .mat files, .jpg, or any other format you like. Ouput dimensions: [[Height, Width, Spectral Depth]]
"""

import torch
import numpy as np
import random
import argparse
import scipy.io


def img_gen(
    image,
    patch_size: int = 32,
    height: int = 620,
    width: int = 620,
    spectral_depth: int = 3,
    num: int = 1,
) -> list:
    """Idea here to extract overlapping patches from images, select a random pixel in that patch, then stitch all such pixels together to form a new image; This will let you generate new images from existing ones for increasing size of dataset."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # convert to pytorch
    image_tensor = torch.from_numpy(image).float()

    # convert image tensor to PyTorch format (HWC to CHW) and move to GPU
    image_tensor = image_tensor.permute(2, 0, 1).to(device)

    tensor_shape = image_tensor.shape
    # print(f"Image Shape: {tensor_shape}")

    stride1 = (tensor_shape[1] - patch_size) // height
    stride2 = (tensor_shape[2] - patch_size) // width
    patches = image_tensor.unfold(1, patch_size, stride1).unfold(2, patch_size, stride2)

    orig_dim = patches.shape
    # print(f"Patches Original Shape: {orig_dim}")
    targ_dim = [spectral_depth, height, width, patch_size, patch_size]
    # print(f"Patches Target Shape: {targ_dim}")

    all_images = []
    for _ in range(num):
        # prepping rows/cols/wavelengths to drop
        wavelengths_to_drop = random.sample(
            range(orig_dim[0]), orig_dim[0] - targ_dim[0]
        )
        rows_to_drop = random.sample(range(orig_dim[1]), orig_dim[1] - targ_dim[1])
        cols_to_drop = random.sample(range(orig_dim[2]), orig_dim[2] - targ_dim[2])

        # slicing tensor
        target_tensor = patches[
            [i for i in range(patches.size(0)) if i not in wavelengths_to_drop],
            :,
            :,
            :,
            :,
        ]
        target_tensor = target_tensor[
            :,
            [j for j in range(target_tensor.size(1)) if j not in rows_to_drop],
            :,
            :,
            :,
        ]
        target_tensor = target_tensor[
            :,
            :,
            [k for k in range(target_tensor.size(2)) if k not in cols_to_drop],
            :,
            :,
        ]

        # reshaping a bit
        reshaped_patches = target_tensor.contiguous().view(
            spectral_depth, -1, patch_size, patch_size
        )
        reshaped_patches_shape = reshaped_patches.shape

        # dropping some rows and cols
        selected_row_indices = [
            random.randint(0, patch_size - 1) for _ in range(reshaped_patches_shape[1])
        ]
        selected_col_indices = [
            random.randint(0, patch_size - 1) for _ in range(reshaped_patches_shape[1])
        ]
        final_image = torch.stack(
            [
                reshaped_patches[:, i, selected_row_indices[i], selected_col_indices[i]]
                for i in range(reshaped_patches_shape[1])
            ]
        )

        # reshaping to get something back to see
        final_image = final_image.transpose(1, 0).reshape(spectral_depth, height, width)

        final_image = (
            final_image.cpu().numpy().transpose(1, 2, 0)
        )  # changing dim to HWC

        all_images.append(final_image)

    return all_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Patch Extractor; output like Frankenstien"
    )
    parser.add_argument(
        "-ps",
        "--patch_size",
        type=int,
        default=32,
        help="Enter patch size to extract from the original image",
    )
    parser.add_argument(
        "-ht", "--height", type=int, default=620, help="Enter height of the final image"
    )
    parser.add_argument(
        "-w", "--width", type=int, default=620, help="Enter width of the final image"
    )
    parser.add_argument(
        "-sd",
        "--spectral_depth",
        type=int,
        default=512,
        help="Enter spectral depth of the final image",
    )
    parser.add_argument(
        "-num", "--num", type=int, default=1, help="Enter number of images to generate"
    )
    parser.add_argument(
        "-file",
        "--file",
        type=str,
        default="Indian_pines.mat",
        help="Enter path to file to use",
    )
    parser.add_argument(
        "-label",
        "--label",
        type=str,
        default="indian_pines",
        help="Enter label to use for .mat file",
    )

    args = parser.parse_args()

    # load image
    image = scipy.io.loadmat(args.file)[args.label]
    image = np.float32(image)

    all_images = img_gen(
        image,
        patch_size=args.patch_size,
        height=args.height,
        width=args.width,
        spectral_depth=args.spectral_depth,
        num=args.num,
    )

    for i in range(len(all_images)):
        scipy.io.savemat(f"patched_image{i}.mat", {f"patched_image{i}": all_images[i]})
        print(f"Patched image saved successfully to patched_image{i}.mat")
