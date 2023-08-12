'''
augment_data.py 

Augments a dataset of hyperspectral datacubes using the CutMix and MixUp methods, and then applies relevant striping artifacts.

Author(s): Ian Vyse,
'''

#IMPORTS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cutmix as cm
import mixup as mu
import apply_stripes

np.random.seed(42)
tf.random.set_seed(42)

#HYPERPARAMETERS
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 0 #TODO: PLACEHOLDER - won't worry about this for now
IMG_SIZE = 512 #Might be 640 - sensor is 512 by 640 so which dimension is which?

#IMPORT
def import_data(filename):
    '''
    Import the Numpy datacube in the .npy file with the desired filename.

    Args:
        filename: path of .npy file containing individual datacube.
    
    Returns:
        datacube: Numpy array containing individual datacube.
    '''
    datacube = np.load("../datasets/" + filename)
    return datacube

#CUTMIX IMPLEMENTATION
'''
TODO:
Key changes:
    1. Hyperspectral data is not labelled. Need to remove labels from the CutMix implementation.
    2. Hyperspectral data is 3D not 2D. Need to apply CutMix to two datasets on one badn and then extend to all bands.
Questions:
    - Hyperparameters?
    - Do we need to split into test and training at this stage?
'''

@tf.function
def modified_cutmix(datacube_one, datacube_two):
    '''
    Perform CutMix on the two hyperspectral datacubes datacube_one and datacube_two.
    Requires that both hyperspectral datasets are the same dimensions.
    Assumes the outermost index of the datacube is the spectral dimension (lambda)

    Args:
        datacube_one: .npy datacube of one hyperspectral dataset
        datacube_two: .npy datacube of another hyperspectral dataset

    Returns:
        datacube: CutMixed product of both datasets
    '''

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    lambda_value = cm.sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = cm.get_box(lambda_value)

    datacube = []

    for i in range(0, len(datacube_one)): #i.e. for each index/spectra in datacube_one
        (image1), (image2) = datacube_one[i], datacube_two[i] #Assuming this line originally selects a single image and it's label
        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        datacube[i] = image1 + image2 #make the ith index of the datacube the CutMix of the ith indices of image1 and image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

    return datacube

#MIXUP IMPLEMENTATION
def modified_mixup(datacube_one, datacube_two, alpha=0.2):
    '''
    
    '''
    # Unpack two datasets
    images_one = datacube_one[0]
    images_two = datacube_two[0]
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = mu.sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))

    datacube = []

    for i in range(0, len(datacube_one)):
    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
        datacube[i] = images_one * x_l + images_two * (1 - x_l)
    
    return datacube

#EVERYTHING ELSE
def apply_stripes():
    '''
    
    '''
    return

def augment_data(filename_one, filename_two):
    '''
    Pairwise dataset augmentation and striping artifact application.

    Args:
        filename_one: filename of 1st dataset
        filename_two: filename of 2nd dataset

    Returns:

    '''

    '''
    TODO:
    Notes about implementation:
        - Maybe we import two or more datasets as a tensor(?) or vector of datacubes
        - That way we can apply stripes to each datacube in that vector at the end
        - The output would be many augmented datasets
    '''

    #Import the two datasets
    dataset_one = import_data(filename_one)
    dataset_two = import_data(filename_two)

    #Apply modified cutmix to the two datasets to produce the augmented dataset
    augmented_dataset = [modified_cutmix(dataset_one, dataset_two), modified_mixup(dataset_one, dataset_two)]

    #TODO: MixUp goes in here somewhere as well.

    #Apply stripes
    number_of_stripes = 20 #TODO: PLACEHOLDER
    augmented_dataset = [apply_stripes.add_multiplicative_stripes(augmented_dataset[0], number_of_stripes), apply_stripes.add_multiplicative_stripes(augmented_dataset[1], number_of_stripes)]

    return augmented_dataset

#MAIN BLOCK
if __name__ == "__main__":
    print("Done")