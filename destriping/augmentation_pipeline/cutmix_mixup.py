import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def cutmix_augmentation(dataset, idx1, idx2, size=16):
    img1 = dataset[idx1]
    img2 = dataset[idx2]

    h, w = img1.shape[0], img1.shape[1]
    x1, y1 = random.randint(0, h - size), random.randint(0, w - size)

    new_img = img1.clone()
    new_img[x1 : x1 + size, y1 : y1 + size] = img2[x1 : x1 + size, y1 : y1 + size]
    return new_img


def mixup_augmentation(dataset, idx1, idx2, alpha=0.5):
    img1 = dataset[idx1]
    img2 = dataset[idx2]

    mixed_img = alpha * img1 + (1 - alpha) * img2
    return mixed_img


def generate_augmented_images(
    dataset, num_samples, augmentation_type="cutmix", alpha=0.5, size=16
):
    augmented_images = []
    for _ in range(num_samples):
        idx1, idx2 = random.sample(range(len(dataset)), 2)

        if augmentation_type == "cutmix":
            augmented_image = cutmix_augmentation(dataset, idx1, idx2, size=size)
        elif augmentation_type == "mixup":
            augmented_image = mixup_augmentation(dataset, idx1, idx2, alpha=alpha)
        else:
            raise ValueError("Unsupported augmentation type")

        augmented_images.append(augmented_image)

    return augmented_images


def visualize_augmentations(dataset, indices, augmentation_function):
    fig, axs = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)))

    for row, idx in enumerate(indices):
        original_image = dataset[idx]
        augmented_image = augmentation_function(
            dataset, idx, random.randint(0, len(dataset) - 1)
        )

        # Convert to PIL Image for visualization
        original_image_pil = transforms.ToPILImage()(original_image)
        augmented_image_pil = transforms.ToPILImage()(augmented_image)

        axs[row, 0].imshow(original_image_pil)
        axs[row, 0].set_title("Original Image")
        axs[row, 0].axis("off")

        axs[row, 1].imshow(augmented_image_pil)
        axs[row, 1].set_title("Augmented Image")
        axs[row, 1].axis("off")

    plt.show()
