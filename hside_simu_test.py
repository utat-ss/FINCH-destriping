import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup import Engine, train_options, make_dataset
import time
from scipy.io import loadmat, savemat
import copy

from matplotlib import pyplot as plt

def add_basic_stripes(data_cube, num_stripes=0):
    """
    Add stripes to original data set which has no stripes. Stripes are added randomly and have no specific patterns

    Args:
    data_cube: orginal data cube
    num_stripes: optional variable, if number is given, the code will generate that amount of stripes on each frame. If number not given, the code will randomly choose the number of stripes for each frame

    Returns:
    striped_data: data cube with stripes added to it
    """

    cube_dim = data_cube.shape  # cube dimesion

    striped_data = copy.deepcopy(
        data_cube
    )  # copying original data cube to not affect changes we make to it here

    if (
        num_stripes == 0
    ):  # random number of stripes for the frame, orignally 25%-75%  are striped changed to 25% to 60% of col are striped
        num_stripes = np.random.randint(
            low=int(0.25 * cube_dim[1]), high=int(0.6 * cube_dim[1]), size=cube_dim[2]
        )

    for i in range(0, cube_dim[2], 1):  # going through each frame
        col_stripes = random.sample(
            range(0, cube_dim[1]), num_stripes[i]
        )  # create a list of non repeating int numbers for size of data cube
        multiplier = (
            np.random.randint(low=5, high=16, size=num_stripes[i]) / 10
        )  # create list of repeating random number, this include number 1, we may want to prevent that

        for k in range(
            0, len(col_stripes), 1
        ):  # go through each column that we will add stripes to and do the multiplier
            striped_data[:, col_stripes[k], i] *= multiplier[k]

    return striped_data

if __name__ == '__main__':


    mat = loadmat("/Users/prithviseran/Documents/UTAT/SERT/icvl_noise/512_mix/KSC.mat")

    #print(mat['KSC'].astype('float64'))

    data = mat['KSC'].astype('float64')

    striped_data = add_basic_stripes(data)

    mat['input'] = striped_data[0:31, 0:31, 0:31]
    mat['gt'] = data[0:31, 0:31, 0:31]

    savemat('second_test.mat', mat)

    mat = loadmat("/Users/prithviseran/Documents/UTAT/SERT/second_test.mat")

    fig, axes = plt.subplots(2,2, figsize=(14,7))
    print("In Shape: ", mat['gt'][:, :, 10].shape, "\n")
    axes[0, 0].pcolormesh(mat['gt'][:, :, 10])
    axes[0, 1].pcolormesh(mat['input'][:, :, 10])
    plt.show()
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.net.use_2dconv)

   
    target_transform = HSI2Tensor()

    """Test-Dev"""

    test_dir = opt.test_dir

    mat_dataset = MatDataFromFolder(
        test_dir) 
    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        ])

    mat_dataset = TransformDataset(mat_dataset, mat_transform)
    mat_loader = DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    )       

    base_lr = opt.lr
    epoch_per_save = 5
    adjust_learning_rate(engine.optimizer, opt.lr)

    engine.epoch  = 0
    

    strart_time = time.time()
    #OVER HERE
    engine.test(mat_loader, test_dir)
    end_time = time.time()
    test_time = end_time-strart_time
    print('cost-time: ',(test_time/len(mat_dataset)))
