import torch
import torchvision.transforms as transforms
import numpy as np


def calculate_psnr(original_images: np.ndarray, compressed_images: np.ndarray,
                   max_pixels: int) -> np.ndarray:
    """Return the ratio of the maximum value of the pixel to noise (MSE) which affects the quality of the pixels for a batch of original and compressed images.

    Parameters:
    - original_images: NumPy array, batch of original image metrics.
    - compressed_images: NumPy array, batch of compressed image metrics.
    - max_pixels: integer, maximum pixel values of the images.

    Return:
        - psnr: Array of PSNR values for each image in the batch.
    """
    mse = np.mean(np.square(original_images - compressed_images),
                  axis=(1, 2, 3))
    psnr = 10 * np.log10((max_pixels ** 2) / mse)
    return psnr


def calculate_ssim(original_images: np.ndarray, compressed_images: np.ndarray,
                   C1: float = 0.01, C2: float = 0.03) -> np.ndarray:
    """Return the SSIM values for a batch of both original and compressed images.

    Parameters:
        - original_images: NumPy array, batch of original image metrics.
        - compressed_images: NumPy array, batch of compressed image metrics.
        - C1: constant value for stability in the formula.
        - C2: constant value for stability in the formula.

    Return:
        - ssim: Array of SSIM values for each image in the batch.
    """
    mu_x = np.mean(original_images, axis=(1, 2, 3))
    mu_y = np.mean(compressed_images, axis=(1, 2, 3))
    sigma_x = np.var(original_images, axis=(1, 2, 3))
    sigma_y = np.var(compressed_images, axis=(1, 2, 3))
    sigma_xy = np.mean(
        (original_images - mu_x[:, np.newaxis, np.newaxis, np.newaxis]) * (
                compressed_images - mu_y[:, np.newaxis, np.newaxis, np.newaxis])
        , axis=(1, 2, 3))

    # sigma_xy holds the covariance between the pixel clarity of both the
    # original and compressed images.

    ssim = (((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) /
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)))
    return ssim


def preprocess_image_batch(images: np.ndarray) -> torch.Tensor:
    """Preprocess a batch of images."""
    preprocess = transforms.Compose([
        # convert imported image into PyTorch tensor.
        transforms.ToTensor(),
        # normalize tensor by subtracting the mean values
        # and dividing by the standard deviations.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # add extra dimension at the start of the tensor and return preprocessed
    # image as PyTorch sensor.
    for image in images:
        preprocessed_images = torch.stack([preprocess(image)])
    return preprocessed_images


def calculate_lpips(original_images: np.ndarray, compressed_images: np.ndarray,
                    perceptual_model) -> np.ndarray:
    """Return the LPIPS distances for a batch of original and compressed images.

    Parameters:
    - original_images: NumPy array, batch of original image metrics.
    - compressed_images: NumPy array, batch of compressed image metrics.
    - perceptual_model: Pre-trained model for perceptual similarity.

    Returns:
    - distances: Array of LPIPS distances for each pair of images in the batch.
    """
    preprocessed_original = preprocess_image_batch(original_images)
    preprocessed_compressed = preprocess_image_batch(compressed_images)

    features_original = perceptual_model(preprocessed_original)
    features_compressed = perceptual_model(preprocessed_compressed)

    distances = torch.nn.functional.pairwise_distance(features_original,
                                                      features_compressed)
    return distances
