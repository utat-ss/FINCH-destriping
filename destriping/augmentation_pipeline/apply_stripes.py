
"""apply_stripes.py.

Contains functions to add stripes to any data set that have absolutely
no stripes

Author(s): Amreen Imrit, John Ma, Hector Chen

Usage: striped_datacube = add_stripes(datacube, configs)

Configs:
snp_noise: bool, true to add salt and pepper noise to hypercube
gaussian_noise: bool, true to add gaussian 'static' noise to hypercube
clusters: bool, true to add stripe clustering
fragmented: bool, true to have stripes fragmented along length
by_layers: bool, true to fragment each spectrum seperately
stripe_frequency: float, frequency of striping
stripe_intensity: float, intensity of striping
salt: float, the probability of salt noise (white pixels) in the image
pepper: float, the probability of pepper noise (black pixels) in the image
max_clusters: int, maximum amount of clusters
bit: int, the bit that the image is in defaulted at 16

"""

import copy
import numpy as np


def add_stripes(datacube, configs=None):
    """
        main function to call
        returns striped datacube based on configs.
    """

    striped_data = copy.deepcopy(
        datacube
    )
    # sets default values if some configs values do not exist
    configs = _check_configs(configs)

    # depending on the stripe types
    striped_data = _gaussian_stripe(striped_data ,configs)

    # add noise to the frame
    if configs['gaussian_noise']:
        striped_data = _add_gaussian_noise(striped_data, configs['bit'])
    if configs['snp_noise']:
        striped_data = _add_snp_noise(striped_data, configs['salt'], configs['pepper'])

    return striped_data


def _check_configs(configs):
    '''
        sets configs to default values if not yet initialized
    '''
    # Define default values
    default_values = {
        'snp_noise': False,
        'gaussian_noise': False,
        'clusters': True,
        'fragmented': True,
        'by_layers': True,
        'stripe_frequency': 0.5,
        'stripe_intensity': -1, 
        'salt':-1, 
        'pepper':-1, 
        'max_clusters': 10,
        'bit': 16
    }
    if configs is None:
        return default_values
    for key, default_value in default_values.items():
        configs[key] = configs.get(key, default_value)
    return configs


def _gaussian_stripe(data, configs):
    '''
        adds solid striped noise of gaussian-varied magnitude to hypercube.

        Reference: https://www.mdpi.com/2072-4292/6/11/11082, Section 2.1
            Noise is additive
            Magnitude: Sensor independent White Gaussian Noise
            0 Mean, 1 Standard Deviation. Rescaled later
            stripe intensity ranges from 0.1%-5% of individual spectrum's range
    '''
    # Setting stripe intensity
    stripe_intensity = configs['stripe_intensity'] # [0, 1)
    if stripe_intensity < 0:
        stripe_intensity = np.random.uniform(0.01, 0.3,1)
    else:
        # making sure each stripe has some variation to it in terms of intensity
        stripe_intensity = max(0, stripe_intensity - 0.05)
        stripe_intensity = np.random.uniform(stripe_intensity, stripe_intensity + 0.1,1)


    # Selecting columns to be striped
    dims = data.shape
    col_lines = _select_lines(dims, configs)

    for i in range(data.shape[2]):

        if configs['by_layers']:
            # Individual spectra are striped seperately
            col_lines = _select_lines(dims, configs)

        # Getting mean and standard deviation
        max_value= np.max(data[:,:,i])
        min_value = np.min(data[:,:,i])
        range_value= max_value - min_value
        mean=0
        std_dev=stripe_intensity*range_value

        # Creating a stripe intensity array corresponding to each stripe
        noise =np.round(abs(np.random.normal(mean, std_dev, len(col_lines)))).astype('<u2')

        # adding noise to data
        if configs['fragmented']:
            # fragments each column
            data = _choose_slices(data, i ,col_lines ,noise)
        else:
            data[:, col_lines, i] += noise
    return data



def _generate_numbers(target_sum, array_size):
    # Generate random numbers between 0 and max_value
    random_numbers = []
    curr = target_sum
    max_value = target_sum
    while len(random_numbers) < array_size:
        temp = np.random.uniform(0, max_value, 1)[0]

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
        num_fragments = np.random.randint(0, data.shape[1]-1, 1)

        fragment_sizes = _generate_numbers(data.shape[1]-1, num_fragments)

        fragment_starts = np.cumsum(fragment_sizes)

        fragment_starts = np.append(0, fragment_starts).astype('<u2')
        

        # random number of fragments
        num_frag_noise = np.random.randint(0, fragment_starts.shape[0])
        
        # get random indices 
        frags = np.array(np.random.choice(fragment_starts.shape[0], num_frag_noise, replace = False))

        # for each fragment add noise to it

        for frag in frags:
            start = fragment_starts[frag]

            # odd that this is an array but ok

            # if at capped size, then add nothing
            if frag < fragment_sizes.shape[0]:
                end = start + fragment_sizes[frag]
            else:
                end = start + 1
            data[start:end, col, i] += noise[j]

            # temp = data[start:end, col, i] # for debugging purpoess
            # temp = temp + noise[j]
            # data[start:end, col, i] = temp # for debugging purposes

    
    return data



def _add_snp_noise(cube, salt_prob, pepper_prob):
    """
    adds salt and pepper noise to hypercube
    """
    if salt_prob < 0:
        salt_prob = np.random.uniform(0, 0.1, 1 )
    if pepper_prob < 0:
        pepper_prob = np.random.uniform(0, 0.1, 1)
    # Salt noise
    salt_mask = np.random.rand(*cube.shape) < salt_prob
    for i in range(cube.shape[2]):
        cube[salt_mask] = 1.0*np.max(cube[:,:,i])
    
    # Pepper noise
    pepper_mask = np.random.rand(*cube.shape) < pepper_prob
    cube[pepper_mask] = 0.0
    
    return cube

def _add_gaussian_noise(cube,bit, mean_percent = 0.05, std_percent=0.1 ):
    """
    adds gaussian 'static' noise to hypercube
    """
    # scales noise by maximum values in each spectrum
    maximums = np.amax(cube, axis=(0, 1))
    gauss = np.random.normal(mean_percent, std_percent, cube.shape)
    for i in range(cube.shape[2]):
        gauss[:,:,i]*=maximums[i]
    noisy_image = cube + gauss
    return np.clip(noisy_image, 0, 2**bit)




def _select_lines(dims, configs):
    """
    helper function that returns a int list of lines to be striped in gaussian striping.
    responsible for clustering stripes
    """

    cols_striped = set()
    max_clusters = configs['max_clusters']
    clusters = configs['clusters']
    stripe_frequency = configs['stripe_frequency']

    # if there are clusters we would want to find areas to cluster these values
    # number of clusters are usual uniform, in addition to the number of stripes
    if clusters:
        # number of clusters
        num_clusters = np.round(np.random.uniform(0, max_clusters, 1)).astype(np.int32)
        # all locations to generate clusters
        cluster_cols = np.round(np.random.uniform(0, dims[1]-1, num_clusters)).astype(np.int32)
        
        # for each cluster center
        for col in cluster_cols:
            # add number of stripes per cluster
            # stripe frequency greater than is an illegal input and will default to random generation
            if (stripe_frequency > 1) or (stripe_frequency < 0):
                num_stripes = np.round(np.random.uniform(0, dims[1], size=1)).astype(np.int32)
            else:
                num_stripes = round(stripe_frequency*(dims[1]-1))

            # for each "cluster center" generate a cluster ie, positions of each striped column
            # may need to change how this works  bc of how intensity works (if intensity high, becomes closer together
            num_stripes = max(0,min(num_stripes, dims[1]-1))

            num_stripes=num_stripes//max_clusters

            start = max(0, col - num_stripes//2)
            end = min(col + num_stripes//2, dims[1] -1)

            
            cluster = np.round(np.random.uniform(start,end, num_stripes)).astype(np.int32)
            cluster = np.clip(cluster, 0, dims[1]-1)
            cols_striped.update(cluster)
        # adding remaining non-cluster stripes
        num_lines =  int((dims[1])*stripe_frequency)
        cols_striped.update(np.round(np.random.uniform(0, dims[1]-1, num_lines - len(cols_striped))).astype(np.int32))
        # make sure each line is non-repeating in the list of lines
        cols_striped = np.array(list(cols_striped)).astype(np.int32)
        # stripe frequency here
    elif  1 < stripe_frequency  or stripe_frequency < 0:
        num_lines =np.round(np.random.uniform(0, (dims[1]-1)//2, 1)).astype(np.int32)
    else:
        num_lines =  int((dims[1])*stripe_frequency)
        cols_striped =np.round(np.random.uniform(0, dims[1]-1, num_lines - 1)).astype(np.int32)
    # print(cols_striped)
    return cols_striped

