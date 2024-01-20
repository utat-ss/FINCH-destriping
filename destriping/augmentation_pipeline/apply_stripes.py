"""apply_stripes.py.

Contains functions to add stripes to any data set that have absolutely
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
        'snp_noise': True,
        'gaussian_noise': True,
        'clusters': True,
        'fragmented': True,
        'by_layers': True,
        'noise_level': -1, # randomize this
        'salt':-1, # randomize
        'peppers':-1, # randomize
        'max_clusters': 10,
        'bit': 16
    }
    if configs is None:
        return default_values
    # Update configs with default values for missing keys
    for key, default_value in default_values.items():
        configs[key] = configs.get(key, default_value)
    return configs

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
    if noise_level < 0:
        noise_level = np.random.uniform(0,0.5,1)


    # select columns
    if configs['by_layers']is True:
        col_lines = _select_lines(dims,configs)


    for i in range(data.shape[2]):
        if configs['by_layers'] is False:
            col_lines = _select_lines(dims,configs)
        
        
        mean=0
        range_value=np.max(data[:,:,i])-np.min(data[:,:,i])
        std_dev=noise_level*range_value


        # all the col lines to add noise to
        noise=np.round(np.random.normal(mean, std_dev, len(col_lines))).astype('<u2')

        # fragment these colines
        
        # choose lengths and fragments
        if configs['fragmented'] is True:
            # for a slice?
            data = _choose_slices(data, i ,col_lines ,noise)
        else:
            data[:,col_lines,i] += noise
    return data

def _generate_numbers(target_sum, max_value, array_size):
    # Generate random numbers between 0 and max_value
    random_numbers = []
    curr = target_sum
    while len(random_numbers) < array_size:
        temp = np.random.uniform(0, max_value, 1)
        if curr >= temp:
            random_numbers.append(temp)
            curr = curr - temp
        else:
            random_numbers.append(curr)
            curr = 0
            break

    return np.round(np.array(random_numbers)).astype('<u2')

def _choose_slices(data, i, col_lines, noise):

    # Number of slices along the specified axis
    for j, col in enumerate(col_lines):
        
        # create a random number of fragments on the column
        num_fragments = np.random.randint(0, data.shape[1], 1)

        fragment_sizes = _generate_numbers(data.shape[1]-1, data.shape[1] -1 , num_fragments)

        fragment_starts = np.cumsum(fragment_sizes)[:-1]
        fragment_starts = np.append(0, fragment_starts).astype('<u2')
        

        # random number of fragments
        num_frag_noise = np.random.randint(0, fragment_starts.shape[0])
        
        # get random indices 
        frags = np.array(np.random.choice(fragment_starts.shape[0], num_frag_noise, replace = False))

        # for each fragment add noise to it
        for frag in frags:
            start = fragment_starts[frag]

            # odd that this is an array but ok
            end = start + fragment_sizes[frag]
            data[start:end[0], col, i] += noise[j]
    
    return data



def _add_snp_noise(cube, salt_prob, pepper_prob):
    """
    args:
    cube: the datacube
    std: the standard deviation of the noise
    mean: the standard mean of the amount that the stripe covers vertically 
    bit: the number of bits the image has

    returns:
    """
    if salt_prob < 0:
        salt_prob = np.random.uniform(0, 0.5, 1 )
    if pepper_prob < 0:
        pepper_prob = np.random.uniform(0, 0.5, 1)
    # Salt noise
    salt_mask = np.random.rand(*cube.shape) < salt_prob
    cube[salt_mask] = 1.0
    
    # Pepper noise
    pepper_mask = np.random.rand(*cube.shape) < pepper_prob
    cube[pepper_mask] = 0.0
    
    return cube

def _add_gaussian_noise(cube,bit, mean_percent = 0.05, std_percent=0.1 ):
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
    max_clusters = configs['max_clusters']
    clusters = configs['clusters']

    #TODO figure out stripe intensity

    # if there are clusters we would want to find areas to cluster these values
    # number of clusters are usual uniform, in addition to the number of stripes
    if clusters:
        # number of clusters
        num_clusters = np.round(np.random.uniform(0, max_clusters, 1)).astype(np.int32)
        # all locations to generate clusters
        cluster_cols = np.round(np.random.uniform(0, dims[1]-1, num_clusters)).astype(np.int32)
        
        # for each cluster center
        for cols in cluster_cols:
            # add number of stripes per cluster
            # max number of stripes must be less than a certain number
            num_stripes = np.round(np.random.uniform(0, dims[1], size=1)//num_clusters).astype(np.int32)
            # for each "pivot" generate a cluster ie, positions of each striped column
            # may need to change how this works  bc of how intensity works (if intensity high, becomes closer together
            num_stripes = np.clip(num_stripes, 0, dims[1]-1)

            cluster = np.round(np.random.uniform(cols - num_stripes//2 ,cols + num_stripes//2 , num_stripes)).astype(np.int32)
            cluster = np.clip(cluster, 0, dims[1]-1)
            cols_striped.update(cluster)
        # make sure each line is non-repeating in the list of lines
        cols_striped = np.array(list(cols_striped))
    else:
        # stripe frequency here
        num_lines =np.round(np.random.uniform(0, (dims[1]-1), 1)).astype(np.int32)
        cols_striped =np.round(np.random.uniform(0, dims[1]-1, num_lines)).astype(np.int32)

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
    _gaussian_stripe(striped_data ,configs)


    # add noise to the frame
    if configs['gaussian_noise']:
        striped_data = _add_gaussian_noise(striped_data, configs['bit'])
    if configs['snp_noise']:
        striped_data = _add_snp_noise(striped_data, configs['salt'], configs['pepper'])

    return striped_data