# stdlib
import os
import random

# external
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset

from ..augmentation_pipeline.apply_stripes import add_stripes


def cutmix_augmentation(cube1, cube2, size=16):
    D, H, W = cube1.shape[-3:]
    zd, yd, xd = (
        random.randint(0, D - size),
        random.randint(0, H - size),
        random.randint(0, W - size),
    )
    cube1[:, zd : zd + size, yd : yd + size, xd : xd + size] = cube2[
        :, zd : zd + size, yd : yd + size, xd : xd + size
    ]
    return cube1


def mixup_augmentation(cube1, cube2, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    mixed_cube = lam * cube1 + (1 - lam) * cube2
    return mixed_cube


class HyperSpectralDataCube(Dataset):
    def __init__(
        self,
        directory,
        label="data",
        train=True,
        percentage_of_empty_cubes=0.6,
        percent_of_scene=(0.01, 0.1),
        val_len=None,
        train_len=None,
    ):
        self.directory = directory
        self.train = train
        self.percentage_of_empty_cubes = percentage_of_empty_cubes
        self.percent_of_scene = percent_of_scene
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".mat")
        ]
        self.dataset_size = train_len if train else val_len
        self.label = label

    def __len__(self):
        return self.dataset_size if self.dataset_size is not None else len(self.files)

    def load_cube(self, file_path):
        data = scipy.io.loadmat(file_path)[self.label]
        cube = np.float32(data)
        return torch.from_numpy(cube).float()

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        cube = self.load_cube(file_path)
        D, H, W = cube.shape

        if D < 32 or H < 32 or W < 32:
            raise ValueError(
                f"Cube dimensions {D}x{H}x{W} are too small for sampling 32x32x32 patches."
            )

        xd, yd, zd = (
            random.randint(0, D - 32),
            random.randint(0, H - 32),
            random.randint(0, W - 32),
        )
        sampled_cube = cube[xd : xd + 32, yd : yd + 32, zd : zd + 32].unsqueeze(
            0
        )  # Add channel dimension

        if self.train and (random.random() < self.percentage_of_empty_cubes):
            percent_of_scene = np.random.uniform(
                self.percent_of_scene[0], self.percent_of_scene[1]
            )
            empty_cube = torch.zeros_like(sampled_cube)
            return {
                "corrupted_input": add_stripes(empty_cube),
                "input": empty_cube,
                "scale": percent_of_scene,
            }

        if random.random() > 0.5:
            other_index = random.randint(0, len(self.files) - 1)
            other_file_path = self.files[other_index]
            other_cube = self.load_cube(other_file_path)
            other_cube = other_cube[xd : xd + 32, yd : yd + 32, zd : zd + 32].unsqueeze(
                0
            )

            if random.random() > 0.5:
                sampled_cube = cutmix_augmentation(sampled_cube, other_cube)
            else:
                sampled_cube = mixup_augmentation(sampled_cube, other_cube)

        percent_of_scene = np.random.uniform(
            self.percent_of_scene[0], self.percent_of_scene[1]
        )
        return {
            "corrupted_input": add_stripes(sampled_cube),
            "input": sampled_cube,
            "scale": percent_of_scene,
        }
