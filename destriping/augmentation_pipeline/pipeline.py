import sys
import os
import shutil
import numpy as np
import ast
import json

sys.path.append("..")
from data.loader import HyperSpectralCube, AugmentedDataset
from apply_stripes import add_stripes
from augmentation_pipeline.cutmix_mixup import generate_augmented_images, visualize_augmentations, cutmix_augmentation, mixup_augmentation
import random
import argparse

def main(args):
    hyper_dataset = HyperSpectralCube(args.hypercube_path, args.hyperspectral_label)
    aug = AugmentedDataset(
        hyper_dataset,
        alpha=args.alpha,
        size=args.size,
        mixup_images=args.mixup_images,
        cutmix_images=args.cutmix_images
    )


    # look for striping configs if it doesnt exist use default
    try:
        with open(args.striping_configs) as f:
            striping_configs = json.load(f)
    except FileNotFoundError:
        striping_configs = None
        print("Striping configs not found. Setting default configs")
   
    aug.images = add_stripes(np.float32(aug.cube), striping_configs)


    output_directory = args.output_directory
    # create a different thing

    #remove existing output directory and create a new one
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    
    for i in range(aug.__len__()):
        output_path = f"{output_directory}/{i + 1}.png"
        aug.plot(i, save_to=output_path)
    print(f"Saved augmented dataset to {output_directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save augmented images from a hypercube.")
    parser.add_argument("--hypercube_path", type=str, help="Path to the hypercube file.")
    parser.add_argument("--hyperspectral_label", type=str, default="paviaU", help="The label of the hyperspectral cube in the .mat file.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for AugmentedDataset.")
    parser.add_argument("--size", type=int, default=128, help="Size parameter for AugmentedDataset.")
    parser.add_argument("--mixup_images", type=int, default=15, help="Number of mixup images for AugmentedDataset.")
    parser.add_argument("--cutmix_images", type=int, default=15, help="Number of cutmix images for AugmentedDataset.")
    parser.add_argument("--output_directory", type=str, default="output_images", help="Directory to save augmented images.")
    parser.add_argument("--striping_configs", type=str, default={},help="Configuration list of apply stripes from a json/text file")

    args = parser.parse_args()
    main(args)
