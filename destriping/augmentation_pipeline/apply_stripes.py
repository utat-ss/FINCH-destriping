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
        'stripe_frequency': 1,
        'noise_range_scaling': 1,
        'max_clusters': 10,
        'bit': 16
    }
    if configs is None:
        return default_values
    # Update configs with default values for missing keys
    for key, default_value in default_values.items():
        configs[key] = configs.get(key, default_value)
    return configs

def _binary_stripe(data, configs):

    '''
    TODO
    REDO THIS:
    Issues:
        - col lines selects the lines we need
        - 
    '''
    # Stripe is there or it isn't.
    dims = data.shape

    noise_range_scaling = configs['noise_range_scaling']

    if configs['vary_len'] is True:
        start_index, end_index = _choose_len_indices(dims)

    # lines to edit
    if configs['by_layer'] is False:
        # same lines for each layer to edit
        # TODO: EDIT SELECT LINES!!!!!!!
        col_lines = _select_lines(dims,configs)

    for i in range(data.shape[2]):
        if configs['by_layer'] is True:
            # selects lines for one layer
            col_lines = _select_lines(dims,configs)
        
        # Generate a random array of indices based on this col lines
        # TODO: change the output of select lines to only output ND-arrays
        random_indices = np.random.choice(len(col_lines), size=len(col_lines), replace=False)

        # Split the array based on the random indices
        part1 = data[random_indices[:len(col_lines)//2]]
        part2 = data[random_indices[len(col_lines)//2:]]

        # set the stripe intensity
        stripe_intensity=(np.max(data[:,:,i])-np.min(data[:,:,i]))*noise_range_scaling

        # Half frequency positive. Half frequency negative.

        # Positive Stripes
        #is_noise=np.random.rand(data.shape[1])< stripe_frequency/2

        noise=np.tile(np.where(part1, stripe_intensity, 0), (data.shape[0],1))

        # Negative Stripes
        #is_noise=np.random.rand(data.shape[1])< stripe_frequency/2

        noise+=np.tile(np.where(part2, -stripe_intensity, 0), (data.shape[0],1))

        data[start_index:end_index,col_lines,i]+=noise

    return data

def _gaussian_stripe(data, configs):
    # Reference: https://www.mdpi.com/2072-4292/6/11/11082, Section 2.1
    # Direction: invariant per scene or within a short time span (Parallel to direction of motion)
        # Noise is additive
    # Magnitude: Sensor independent White Gaussian Noise (0 mean and specific standard deviations)
        # Dark current varies Gaussianly
        #  0 Mean, 1 Standard Deviation. Rescaled later
        # four different striping scenarios, scales are standard deviations of 0.1%, 0.5%, 1% and 5% of individual band's range
        # pick a random length
    dims = data.shape
    noise_level = configs['noise_level'] # [0, 1)

    if configs['vary_len'] is True:
        start_index, end_index = _choose_len_indices(dims)

    # select columns
    if configs['by_layer']is True:
        col_lines = _select_lines(dims,configs)


    for i in range(data.shape[2]):
        if configs['by_layer'] is False:
            col_lines = _select_lines(dims,configs)

        mean=0
        range_value=np.max(data[:,:,i])-np.min(data[:,:,i])
        std_dev=noise_level*range_value

        noise=np.random.normal(mean, std_dev, len(col_lines))
        data[start_index:end_index,col_lines,i]+= noise
    return data

def _choose_len_indices(dims):
    # pick a random length
    length = np.random.uniform(0, dims[1])

    # pick the center of the stripe
    position = np.random.uniform(0, dims[1])

    # Calculate the start and end indices of the selected portion
    start_index = int(max(position - length / 2, 0))
    end_index = int(min(position + length / 2, dims[0]))
    return start_index, end_index


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




def _select_lines(dims, configs):
    """
    Helper function, 
    Selects lines to add noise to
    args:
    dims: tuple of the data cube.
    clusters: boolean
    r: ratio of columns to select
    max_clusters: int
    v: stripe intensity???

    returns:
    List of int, representing the lines to return
    """


    cols_striped = set()
    max_clusters = configs['max_cluster']
    clusters = configs['clusters']

    #TODO figure out stripe intensity

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
            # max number of stripes must be less than a certain number
            num_stripes = np.round(np.random.uniform(0, dims[1], size=1)//num_clusters).astype(np.int64)
            # for each "pivot" generate a cluster ie, positions of each striped column
            # may need to change how this works  bc of how intensity works (if intensity high, becomes closer together
            num_stripes = np.clip(num_stripes, 0, dims[1]-1)

            cluster = np.round(np.random.uniform(cols - num_stripes//2 ,cols + num_stripes//2 , num_stripes)).astype(np.int64)
            cluster = np.clip(cluster, 0, dims[1]-1)
            cols_striped.add(cluster)
        # make sure each line is non-repeating in the list of lines
        cols_striped = np.array(list(set(cols_striped)))
    else:
        # stripe frequency here
        num_lines =np.round(np.random.uniform(0, (dims[1]-1), 1)).astype(np.int64)
        cols_striped =np.round(np.random.uniform(0, dims[1]-1, num_lines)).astype(np.int64)

    return cols_striped


def add_stripes(datacube, configs=None):
    """
    Adds stripes to the data cube depending on the parameters

    Args:
    datacube: np array
    configs: dictionary
    
    Returns:
    striped_data: data cube with stripes added to it
    """

    striped_data = copy.deepcopy(
        datacube
    )
    # sets default values if some configs values do not exist
    configs = _check_configs(configs)

    # depending on the stripe types
    if configs['stripe_type'] == 'gaussian':
        _gaussian_stripe(striped_data, 0 ,configs)
    elif configs['striped_type'] == 'binary':
        _binary_stripe(striped_data, 1, 1 ,configs)


    # add noise to the frame
    if configs['gauss_noise']:
        striped_data = _add_gaussian_noise(striped_data, configs['bit'])
    if configs['snp_noise']:
        striped_data = _add_snp_noise(striped_data, configs['bit'])

    return striped_data