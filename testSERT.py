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
from email.mime import base, image
from locale import normalize
from math import fabs
from xml.sax import SAXException
import torch
import torch.optim as optim
import models

import os
import argparse

from os.path import join
from utility import *
from utility.ssim import SSIMLoss,SAMLoss
from thop import profile
from torchstat import stat 
import scipy.io as scio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models as torchmodel
from torch import  einsum

import torchvision.utils as vutil
import glob
import time
import cv2

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


def numpy_dataset():
    filelist = glob.glob('./Numpy-data/.npy-files/*.npy')

    images = []

    for name in filelist:
        images.append(np.load(name))

    return images




if __name__ == "__main__":

    #print(len(numpy_dataset()))

    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)

    #Setup Engine
    engine = Engine(opt)

    test_dataset = numpy_dataset()

    test_data_transposed = []
    tensor_with_stripes = []
    targets = []

    new_height, new_width = 160, 160

    for i in range(len(test_dataset)):
        #print(test_dataset[i].shape)
        #print(type(test_dataset[i]))
        test_dataset[i] = np.array(test_dataset[i]).astype(float)
        test = add_basic_stripes(test_dataset[i])
        target = test_dataset[i]
        test = cv2.resize(test, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        test = np.transpose(test, (2, 0, 1))[:31]
        target = np.transpose(target, (2, 0, 1))[:31]

        print(test.shape)

        tensor_with_stripes.append(test)
        targets.append(target)

       
    """
    for i in range(len(test_data_transposed)):
        test = add_basic_stripes(test_data_transposed[i])[:31]
        target = test_data_transposed[i][:31]


        test = np.array(test)[:, 0:145, 0:145]
        target = np.array(target)[:, 0:145, 0:145]

        #HERE                         
        res = cv2.resize(test, dsize=(1300, 1392), interpolation=cv2.INTER_CUBIC)

        print(res.shape)

        #print(test[:, 0:145, 0:145].shape)
        tensor_with_stripes.append(test)
        targets.append(target)

        print(tensor_with_stripes[i].shape)
    """
    tensor_with_stripes = np.array(tensor_with_stripes)
    targets = np.array(targets)

    tensor_with_stripes = torch.from_numpy(tensor_with_stripes).float()
    targets = torch.from_numpy(targets).float()

    outputs, loss = engine.test_numpy(tensor_with_stripes, targets)

    psnr = np.mean(cal_bwpsnr(outputs, targets))

    outputs_numpy = outputs.detach().numpy()

    print("\nPSNR: " , psnr, "\n")
    print(targets.shape)

    fig, axes = plt.subplots(2,2, figsize=(14,7))

    axes[0, 0].pcolormesh(tensor_with_stripes[1][30])
    axes[0, 1].pcolormesh(targets[1][30])
    axes[1, 1].pcolormesh(outputs_numpy[1][30])

    plt.show()



    """    
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)


    #Setup Engine
    engine = Engine(opt)

    test_set = np.random.rand(1, 31, 96, 96)

    tensor1 = torch.from_numpy(test_set)

    test_data = np.load('/Users/prithviseran/Documents/UTAT/SERT/Numpy-data/.npy-files/indian_pine_array.npy')

    print(test_data.shape)

    #test_data = np.resize(test_data, (200, 145, 145))

    test_data_transposed = np.transpose(test_data, (2, 0, 1))

    print(test_data_transposed.shape)

    tensor1 = torch.from_numpy(np.array([test_data_transposed.astype(float)]))

    tensor1 = tensor1[:,0:31]

    #tensor1 = tensor1[:,0:96, 0:96]

    print(tensor1.shape)



    #fig, axes = plt.subplots(2,2, figsize=(14,7))

    #axes[0, 0].pcolormesh(tensor1[0][0])
    #axes[0, 1].pcolormesh(mat['input'][:, :, 10])
    #plt.show()
    fig, axes = plt.subplots(2,2, figsize=(14,7))

    tensor_float = tensor1.float()

    tensor_with_stripes = add_basic_stripes(tensor_float)

    outputs, loss = engine.test_numpy(tensor_with_stripes, tensor_float)

    outputs_numpy = outputs.detach().numpy()

    print(loss)

    axes[0, 0].pcolormesh(tensor_with_stripes[0][20])
    axes[0, 1].pcolormesh(tensor_float[0][20])
    axes[1, 1].pcolormesh(outputs_numpy[0][20])

    plt.show()

    """
