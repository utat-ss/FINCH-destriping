from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io
import torch
from spectral import imshow, save_rgb
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from augmentation_pipeline.cutmix_mixup import generate_augmented_images


class AugmentedDataset(Dataset):
    def __init__(
        self, dataset, alpha=0.5, size=16, mixup_images=None, cutmix_images=None
    ):
        augmented_images_cutmix = generate_augmented_images(
            dataset, num_samples=cutmix_images, augmentation_type="cutmix", size = size
        )
        augmented_images_mixup = generate_augmented_images(
            dataset, num_samples=mixup_images, augmentation_type="mixup", alpha = alpha
        )
        original_images = [dataset[i] for i in range(dataset.__len__())]
        stacked_tensor = torch.stack(
            # augmented_images_cutmix + augmented_images_mixup
            original_images + augmented_images_cutmix + augmented_images_mixup
        )
        self.images = stacked_tensor
        self.cube = self.images.numpy()

    def __getitem__(self, index):
        x = self.images[index]
        return x

    def __len__(self):
        return len(self.images)

    def plot(self, index, save_to=None):
        plt.imshow(self.images[index])
        if save_to:
            plt.savefig(save_to)
        # plt.show()
        return


class HyperSpectralCube(Dataset):
    def __init__(self, mat_path, label="paviaU"):
        data = scipy.io.loadmat(mat_path)[label]
        data = np.float32(data)
        self.cube = data
        data = np.moveaxis(data, -1, 0)
        self.images = torch.from_numpy(data)
        self.bands = len(self.images)
        
    def __getcube__(self):
        return self.images

    def __getitem__(self, band):
        x = self.images[band]
        return x

    def __len__(self):
        return self.bands

    def plot(self, band, save_to = None, show = False):
        imshow(self.cube, [band])
        if save_to:
            save_rgb(save_to, self.cube, [band])
            if show:
                plt.imshow(self.images[band])
                plt.show()
        return