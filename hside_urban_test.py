import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup import Engine, train_options, make_dataset
from hside_simu_test import add_basic_stripes
import numpy as np


# python hside_urban_test.py -a sert_urban -p sert_urban_test -r -rp ./checkpoints/real_urban.pth

if __name__ == '__main__':

    mat = loadmat("/Users/prithviseran/Documents/UTAT/SERT/icvl_noise/512_mix/KSC.mat")

    #print(mat['KSC'].astype('float64'))

    data = mat['KSC'].astype('float64')

    striped_data = add_basic_stripes(data)

    print(striped_data.shape)

    b = np.random.randn(512, 614, 34)

    striped_data = np.concatenate((striped_data, b), axis=2)

    striped_data = np.transpose(striped_data, (2, 0, 1))

    print(striped_data.shape)

    mat['input'] = striped_data #[0:31, 0:31, 0:31]
    #mat['gt'] = data[0:31, 0:31, 0:31]

    savemat('urban.mat', mat)
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
    basefolder = '/Users/prithviseran/Documents/UTAT/SERT/data/'
    
    
    mat_datasets = [MatDataFromFolder(
        basefolder, size=1,fns=['urban.mat']) ]

    
    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='input',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='input', needsigma=False),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

 
    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]        

    
    engine.test(mat_loaders[0], basefolder)
        
