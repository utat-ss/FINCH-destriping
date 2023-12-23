"""apply_stripes.py.

Contains fucntions to add stripes to any data set that have absolutely
no stripes

Author(s): Amreen Imrit, John Ma

"""

# stdlib
import copy
import random

# external
import numpy as np


# Creating a class to stripe a datacube
# Helps with data

def _vary_len(dims, col):
    """
    Takes a column of a frame of the data cube and 
    makes a varied length for it.
    args: 

    returns:

    """
    # random length and random position
    # we can pick where it can be
    length = random.uniform(0, dims[1])
    position = random.uniform(length/2, dims[1]-length/2)
    # change to a uniform stripe
    for i in position:
        col[i] = 0

    return col


def _add_noise(frame, mean =0, std=25):
    """
    args:
    std: the standard deviation of the noise
    mean: the standard mean of the amount that the stripe covers vertically 

    returns:
    """
    row, col  = frame.shape
    gauss = np.random.normal(mean, std, (row, col))
    noisy_image = frame + gauss
    return np.clip(noisy_image, 0, 255)


def _add_empty(column):
    """
    TODO
    args:

    returns:
    """
    # we can pick a uniform random number between 0-60% of the thing with stripes
    # l = column.shape(0)
    # for i in range(0, column):
    #     column

    return column


def _select_lines(dims, clusters, num_lines, max_clusters):
    """
        args:
        dims: tuple of the data cube.
        clusters: boolean
        num_lines: number of lines to select

        returns:
        List of int, representing the lines to return
    """


    cols_striped = []

    # if there are clusters we would want to find areas to cluster these values
    if clusters:
        num_clusters = int(random.uniform(0, max_clusters))
        # all locations to generate clusters
        cluster_cols = random.sample(range(0, dims[1]), num_clusters)

        for cols in cluster_cols:
            # add number of stripes per cluster
            num_stripes = int(random.gauss(dims[1]/10, dims[1]/10))
            cluster = [np.random.normal(cols, dims[1]/20, num_stripes)]
            cols_striped += cluster
        # make sure each line is non-repeating in the list of lines
        cols_striped = list(set(cols_striped))
    else:
        cols_striped = random.sample((0, dims[1]), num_lines)

    return cols_striped


def add_stripes(datacube, noise=True, empty=True, clusters=True, vary_len=True, mean_stripes = None, var_stripes = None, by_layers = False, max_clusters = 6):
    """
    Adds stripes to the data cube depending on the parameters

    Args:
    noise: boolean, if True add in noise to the datacube
    vary_len: boolean, if True adds multiplicative stripes into the data
    empty: boolean, if True adds empty strips into the data
    clusters: boolean, if True adds clustered stripes into the data
    mean_stripes: float/int,
    var_stripes: float/int, 
    max_clusters: int,
    
    Returns:
    striped_data: data cube with stripes added to it
    """
    data_cube = datacube

    striped_data = copy.deepcopy(
        data_cube
    )
    dims = data_cube.shape

    # default values for number of stripes
    if mean_stripes is None:
        mean_stripes = dims[1]/2
    if var_stripes is None:
        var_stripes = dims[1]/10

    if by_layers is True:
        for i in range(0, dims[2]):
            num_stripes = int(random.gauss(mean_stripes, var_stripes))
            col_lines = _select_lines(dims, clusters, num_stripes, max_clusters) # select a random amount of lines
            for col_line in col_lines:
                striped_col = striped_data[: , col_line, i]
                # vary the length of this column

                if empty and vary_len:
                    _vary_len(dims, striped_col)
                    _add_empty(striped_data, col_line)
                elif vary_len:
                    _vary_len(dims, striped_col)
                #vary the length of this column
                elif empty:
                    _add_empty(striped_data, col_line)

                striped_data[: , col_line, i] = striped_col
            # add noise to the frame
            if noise:
                _add_noise(striped_data[:,:,i])

    elif by_layers is False:
        num_stripes = int(random.gauss(mean_stripes, var_stripes))
        col_lines = _select_lines(dims, clusters, num_stripes, max_clusters) # select a random amount of lines
        for col_line in col_lines:
            striped_col = striped_data[: , col_line, :]
            # vary the length of this column

            if empty and vary_len:
                _vary_len(dims, striped_col)
                _add_empty(striped_data, col_line)
            elif vary_len:
                _vary_len(dims, striped_col)
            #vary the length of this column
            elif empty:
                _add_empty(striped_data, col_line)

            striped_data[: , col_line, :] = striped_col
        # add noise to the frame
        if noise:
            _add_noise(striped_data[:,:,i])

    return striped_data



