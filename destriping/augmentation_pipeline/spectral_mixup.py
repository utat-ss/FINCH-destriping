import numpy as np
import torch


def spectral_mixup(gt, lam, n, device="cuda"):
    fsize, hsize, wsize = gt.shape
    x_mixups, lms_mixups, gt_mixups = [], [], []
    for mixid in range(n):
        conversion_matrix_rand = np.random.rand(fsize, fsize)
        conversion_matrix_rand = np.array(
            conversion_matrix_rand / (np.mean(np.sum(conversion_matrix_rand, axis=0))),
            dtype=np.float32,
        )
        conversion_matrix_rand = torch.from_numpy(conversion_matrix_rand)
        conversion_matrix_rand = conversion_matrix_rand.to(device)

        x_mixup, lms_mixup, gt_mixup = (
            (x + conversion(x, conversion_matrix_rand)) / 2,
            (lms + conversion(lms, conversion_matrix_rand)) / 2,
            (gt + conversion(gt, conversion_matrix_rand)) / 2,
        )
        x_mixups.append(x_mixup)
        lms_mixups.append(lms_mixup)
        gt_mixups.append(gt_mixup)
    return x_mixups, lms_mixups, gt_mixups
