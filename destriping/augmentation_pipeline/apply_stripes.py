"""apply_stripes.py.

Contains fucntions to add stripes to any data set that have absolutely
no stripes

Author(s): Amreen Imrit, John Ma, Hector Chen

"""

# stdlib
import copy

# external
import numpy as np


def _check_configs(configs):
    '''
        checks configs to see if there any empty ones, if so
        fills them with these default values
    '''
    # Define default values
    default_values = {
        'stripe_type': 'gaussian',
        'snp_noise': False,
        'gaussian_noise': False,
        'clusters': False,
        'vary_len': False,
        'by_layers': True,
        'stripe_intensity': 0.7,
        'max_clusters': 10,
        'bit': 16
    }
    if configs is None:
        return default_values
    # Update config with default values for missing keys
    for key, default_value in default_values.items():
        configs[key] = configs.get(key, default_value)
    return configs

def _binary_stripe(data, stripe_frequency, noise_range_scaling):
    # Stripe is there or it isn't.
    for i in range(data.shape[2]):
        stripe_intensity=(np.max(data[:,:,i])-np.min(data[:,:,i]))*noise_range_scaling
        # Half frequency positive. Half frequency negative.

        # Positive Stripes
        is_noise=np.random.rand(data.shape[1])<stripe_frequency/2
        noise=np.tile(np.where(is_noise, stripe_intensity, 0), (data.shape[0],1))

        # Negative Stripes
        is_noise=np.random.rand(data.shape[1])<stripe_frequency/2
        noise+=np.tile(np.where(is_noise, -stripe_intensity, 0), (data.shape[0],1))
        data[:,:,i]+=noise

    return data

def _gaussian_stripe(data, noise_level):
    # Reference: https://www.mdpi.com/2072-4292/6/11/11082, Section 2.1
    # Direction: invariant per scene or within a short time span (Parallel to direction of motion)
        # Noise is additive
    # Magnitude: Sensor independent White Gaussian Noise (0 mean and specific standard deviations)
        # Dark current varies Gaussianly
        #  0 Mean, 1 Standard Deviation. Rescaled later
        # four different striping scenarios, scales are standard deviations of 0.1%, 0.5%, 1% and 5% of individual band's range
    for i in range(data.shape[2]):
        mean=0
        range_value=np.max(data[:,:,i])-np.min(data[:,:,i])
        std_dev=np.array([0.001, 0.005, 0.01, 0.05])*range_value
        # Choose std_dev
        deviation=std_dev[noise_level]
        noise=np.random.normal(mean, deviation, data.shape[1])
        noise=np.tile(noise, (data.shape[0],1))
        data[:,:,i]+=noise
    return data

def _vary_len(striped_data, col_striped, stripe_type, stripe_intensity):
    """
    Helper function
    Takes a column of a frame of the data cube and 
    makes a varied length for it.
    
    Two cases: one for which we get a 240x240 frame
    and the other for the entire data_cube

    args: 
        striped_data: the 3D hyperspectral cube or one frame of it
        col_striped: the column to change

    returns:
        cube
    """
    # random length and random position
    # we can pick where it can be
    dims = striped_data.shape


    
    # pick a random length
    length = np.random.uniform(0, dims[1])

    # pick the center of the stripe
    position = np.random.uniform(0, dims[1])

    # Calculate the start and end indices of the selected portion
    start_index = int(max(position - length / 2, 0))
    end_index = int(min(position + length / 2, dims[0]))


    return

def _add_snp_noise(cube, salt_prob = 0.02, pepper_prob=0.02):
    """
    args:
    cube: the datacube
    std: the standard deviation of the noise
    mean: the standard mean of the amount that the stripe covers vertically 
    bit: the number of bits the image has

    returns:
    """

    # Salt noise
    salt_mask = np.random.rand(*cube.shape) < salt_prob
    cube[salt_mask] = 1.0
    
    # Pepper noise
    pepper_mask = np.random.rand(*cube.shape) < pepper_prob
    cube[pepper_mask] = 0.0
    
    return cube

def _add_gaussian_noise(cube,bit, mean_percent =0.05, std_percent=0.1 ):
    """
    args:
    cube: the datacube
    std: the standard deviation of the noise
    mean: the standard mean of the amount that the stripe covers vertically 
    bit: the number of bits the image has

    returns:
    """
    #max
    _, col, _ = cube.shape
    gauss = np.random.normal(mean_percent*col, std_percent*col, cube.shape)
    noisy_image = cube + gauss
    return np.clip(noisy_image, 0, 2**bit)




def _select_lines(dims, clusters, r, max_clusters, v = 0.05):
    """
    args:
    dims: tuple of the data cube.
    clusters: boolean
    r: ratio of columns to select

    returns:
    List of int, representing the lines to return
    """


    cols_striped = set()

    # if there are clusters we would want to find areas to cluster these values
    # number of clusters are usual uniform, in addition to the number of stripes
    if clusters:
        # number of clusters
        num_clusters = np.round(np.random.uniform(0, max_clusters, 1))
        # all locations to generate clusters
        cluster_cols = np.round(np.random.uniform(0, dims[1]-1, num_clusters)).astype(np.int64)
        
        # for each cluster center
        for cols in cluster_cols:
            # add number of stripes per cluster
            num_stripes = np.round(v*np.random.uniform(0, dims[1]*r, size=1)//num_clusters).astype(np.int64)
            # for each "pivot" generate a cluster ie, positions of each striped column
            # may need to change how this works  bc of how intensity works (if intensity high, becomes closer together
            cluster = np.round(np.random.uniform(cols - num_stripes//2 ,cols + num_stripes//2 , num_stripes)).astype(np.int64)
            cluster = np.clip(cluster, 0, dims[1]-1)
            cols_striped.add(cluster)
        # make sure each line is non-repeating in the list of lines
        cols_striped = set(cols_striped)
    else:
        num_lines =np.round(np.random.uniform(0, (dims[1]-1)*r, 1)).astype(np.int64)
        cols_striped =np.round(np.random.uniform(0, dims[1]-1, num_lines)).astype(np.int64)

    return cols_striped


def add_stripes(datacube, config=None):
    """
    Adds stripes to the data cube depending on the parameters

    Args:
    gauss_noise: boolean, if True add in noise to the datacube
    snp_noise: boolean, if True add in salt and pepper noise
    vary_len: boolean, if True adds multiplicative stripes into the data
    empty: boolean, if True adds empty strips into the data
    clusters: boolean, if True adds clustered stripes into the data
    max_clusters: int, max number of clusters to generate
    bit: int, the number of bits the data is represented in
    
    Returns:
    striped_data: data cube with stripes added to it
    """
    data_cube = datacube

    striped_data = copy.deepcopy(
        data_cube
    )
    dims = data_cube.shape


    config = _check_configs(config)

    # each layer selects different lines
    if config['by_layers'] is True:
        for i in range(0, dims[2]):
            # select a random amount of lines
            col_lines = _select_lines(dims,config['stripe_type'] ,config['clusters'], config['r'], config['max_clusters'])
            # vary the length of these columns
            if config['vary_len']:
                _vary_len(striped_data, col_lines, config['stripe_type'], config['stripe_intensity'])
            else:
                striped_data[:, col_lines, i] = 0


    # each layer selects the same lines
    elif config['by_layers'] is False:
        # select a random amount of lines
        col_lines = _select_lines(dims, config['clusters'], config['r'], config['max_clusters'])
        # vary the length of these columns
        if config['vary_len']:
            _vary_len(striped_data, col_lines, config['stripe_type'], config['stripe_intensity'])
        elif config['stripe_type'] == 'gaussian':
            _gaussian_stripe(striped_data, 0)
        elif config['striped_type'] == 'binary':
            _binary_stripe(striped_data, 1, 1)
            

    # add noise to the frame
    if config['gauss_noise']:
        striped_data = _add_gaussian_noise(striped_data, config['bit'])
    if config['snp_noise']:
        striped_data = _add_snp_noise(striped_data, config['bit'])

    return striped_data