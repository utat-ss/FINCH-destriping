import torch
import torchvision.transforms as transforms
import numpy as np


def calculate_psnr(original_image: np.ndarray, compressed_image: np.ndarray,
                   max_pixels: int) -> float:
    """Return the ratio of the maximum value of the pixel to the noise (MSE)
    that affects the quality of the pixels between two images.

    Parameters:
    - original_image: NumPy array, original image metrics.
    - compressed_image: NumPy array, compressed image metrics.
    - max_pixels: integer, maximum pixel values of the images.

    Return:
        - psnr: PSNR value.
    """
    mse = np.mean((original_image - compressed_image) ** 2)
    psnr = 10 * np.log10((max_pixels ** 2) / mse)
    return psnr


def calculate_ssim(original_image: np.ndarray, compressed_image: np.ndarray,
                   C1: float = 0.01, C2: float = 0.03) -> float:
    """Return the Structural Similarity Index between two images.

    Parameters:
        - original_image: NumPy array, original image metrics.
        - compressed_image: NumPy array, compressed image metrics.
        - C1: constant value for stability in the formula.
        - C2: constant value for stability in the formula.

    Return:
        - ssim: SSIM value.
    """
    mu_x = np.mean(original_image)
    mu_y = np.mean(compressed_image)
    sigma_x = np.var(original_image)
    sigma_y = np.var(compressed_image)
    sigma_xy = np.cov(original_image, compressed_image)[0, 1]
    # sigma_xy holds the covariance between the pixel clarity of both the
    # original and compressed images.

    ssim = (((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) /
            ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)))
    return ssim


def preprocess_image(image):
    """Preprocess the image."""
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
    return preprocess(image).unsqueeze(0)


def calculate_distance(feature1, feature2):
    """Calculate the distance between image features."""
    # compute the pairwise distance between the two feature vectors using
    # the Euclidean distance metric.
    return torch.nn.functional.pairwise_distance(feature1, feature2)


def calculate_lpips(original_image: np.ndarray, compressed_image: np.ndarray,
                    perceptual_model) -> float:
    """Return the Learned Perceptual Image Patch Similarity (LPIPS) between two
    images.

    Parameters:
    - original_image: NumPy array, original image metrics.
    - compressed_image: NumPy array, compressed image metrics.
    - perceptual_model: Pre-trained model for perceptual similarity.

    Returns:
    - distance: LPIPS distance.
    """
    preprocessed_original = preprocess_image(original_image)
    preprocessed_compressed = preprocess_image(compressed_image)

    feature_original = perceptual_model(preprocessed_original)
    feature_compressed = perceptual_model(preprocessed_compressed)

    distance = calculate_distance(feature_original, feature_compressed)
    return distance

