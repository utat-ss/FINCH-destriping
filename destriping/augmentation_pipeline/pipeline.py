import sys

sys.path.append("..")
from data.loader import HyperSpectralCube, AugmentedDataset
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

    output_directory = args.output_directory
    for i in range(aug.__len__()):
        output_path = f"{output_directory}/{i}.png"
        aug.plot(i, save_to=output_path)
    print(f"Saved augmented dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save augmented images from a hypercube.")
    parser.add_argument("--hypercube_path", type=str, help="Path to the hypercube file.")
    parser.add_argument("--hyperspectral_label", type=str, default="paviaU", help="The label of the hyperspectral cube in the .mat file.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for AugmentedDataset.")
    parser.add_argument("--size", type=int, default=32, help="Size parameter for AugmentedDataset.")
    parser.add_argument("--mixup_images", type=int, default=100, help="Number of mixup images for AugmentedDataset.")
    parser.add_argument("--cutmix_images", type=int, default=100, help="Number of cutmix images for AugmentedDataset.")
    parser.add_argument("--output_directory", type=str, default="output_images", help="Directory to save augmented images.")

    args = parser.parse_args()
    main(args)