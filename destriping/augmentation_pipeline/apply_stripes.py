"""apply_stripes.py.

Contains fucntions to add stripes to any data set that have abosolutely
no stripes

Author(s): Amreen Imrit

"""

# stdlib
import copy
import random

# external
import numpy as np

# Creating a class to stripe a datacube
# Helps with data


class Striper:
    '''
       Args
    '''
    def __init__(self, data_cube):
        '''
        Args:
        data_cube: orginal data cube, where dimention is of form [spatial X spatial X spectral]
        '''

        self.data_cube = data_cube

        self.striped_data = copy.deepcopy(
            self.data_cube
        )

        self.cube_dim = self.data_cube.shape  # cube dimesion so i dont have to type this entire thing
        self.lines = None
        self.mean_num_stripes = self.cube_dim[1]/2 # default values for number of stripes
        self.var_num_stripes = self.cube_dim[1]/10
        self.num_stripes = int(random.gauss(
            self.mean_num_stripes, self.var_num_stripes))

    def change_num_stripes(self, mean_num_stripes, var_num_stripes):
        '''
        Generates a random number of stripes based on mean and variance
        Args:
        mean_num_stripes: float, mean number of stripes 
        var_num_stripes: float, variance of the mean number of stripes
        '''
        self.mean_num_stripes = mean_num_stripes
        self.var_num_stripes = var_num_stripes
        self.num_stripes = int(random.gauss(
            self.mean_num_stripes, self.var_num_stripes))

    def set_num_stripes(self, num_stripes):
        '''
        Sets the number of stripes wanted on the data cube
        Args: 
        num_stripes: int, number of stripes
        '''
        self.num_stripes = num_stripes

    def add_stripes(self, noise=True, empty=True, clusters=True, vary_len=True):
        """
        Adds stripes to the data cube depending on the parameters
        
        Args:
        noise: boolean, if True add in noise to the datacube
        mult: boolean, if True adds multiplicative stripes into the data
        empty: boolean, if True adds empty strips into the data
        clusters: boolean, if True adds clustered stripes into the data 

        Returns:
        striped_data: data cube with stripes added to it
        """

        for i in range(0, self.cube_dim[2]):
            col_lines = self._select_lines(clusters)
            for col_line in col_lines:
                striped_col = self.striped_data[1][col_line]
                if vary_len:
                    striped_col = self._vary_len(striped_col)
                if empty:
                    # 0 empty, 1 stays the same
                    self._add_empty()
                self.striped_data[col_line ,:, i] = 0
            # add noise to the frame
            if noise:
                self._add_noise()
    
        return self.lines

    def _add_multiplicative_stripes(self, lines, stripe_per_image=0):
        """Add multiplicative stripes to original data set which has no stripes.
        Stripes are added randomly and have no specific patterns.

        Args:
        lines: Array of lines to add these stripes to
        stripe_per_image: optional variable, if number is given, the code will generate that amount of stripes on each frame. If number not given, the code will randomly choose the number of stripes for each frame

        Returns:
        striped_data: data cube with stripes added to it

        """
        num_stripes = 10
        for i in range(0, self.cube_dim[2], 1):  # going through each frame
            # create a list of non repeating int numbers for size of data cube, choosing which columns will be striped
            col_stripes = random.sample(
                range(0, self.cube_dim[1]), num_stripes)

            # create list of repeating random number, multiplier is choosen b/w 0.5 to 1.6
            multiplier = np.random.randint(
                low=5, high=16, size=num_stripes) / 10

            # go through each column that we will add stripes and mutiply the column values by the multiplier
            for k in range(0, len(col_stripes), 1):
                self.striped_data[:, col_stripes[k], i] *= multiplier[k]

        return self.striped_data

    def _vary_len(self, col):
        """
        args: 

        returns:

        """
        # random length and random position
        # we can pick where it can be
        length = random.uniform(0, self.cube_dim[1])
        position = random.uniform(length/2, self.cube_dim[1]-length/2)

        # change to a uniform stripe

        return col


    def _add_noise(self):
        """
        args:
        std: the standard deviation of the noise
        mean: the standard mean of the amount that the stripe covers vertically 

        returns:
        """
        # noise to the stripes value of the channels to
        # create a gaussian thing
        self.data_cube = 0

        return

    def _add_empty_stripes(self):
        """
        args:

        returns:
        """
        # we can pick a uniform random number between 0-60% of the thing with stripes
        return

    def _select_lines(self, clusters):
        """
            args:
            data_cube: orginal data cube, where dimention is of form [spatial X spatial X spectral]

            returns:
            List of positions of cols to stripe
        """
        cluster_max = 10  # say there are up to 10 clusters max
        cols_striped = []

        # if there are clusters we would want to find areas to cluster these values
        if clusters:
            num_clusters = int(random.uniform(0, cluster_max))
            # all locations to generate clusters
            cluster_cols = random.sample((0, self.cube_dim[1]), num_clusters)
            for cols in cluster_cols:
                # add number of stripes per cluster
                num_stripes = int(random.gauss(
                    self.cube_dim[1]/10, self.cube_dim[1]/10))
                cluster = [np.random.normal(
                    cols, self.cube_dim[1]/20, num_stripes)]
                cols_striped += cluster
            cols_striped = list(set(cols_striped))
        else:
            num_stripes = int(random.gauss(
                self.mean_num_stripes, self.var_num_stripes))
            cols_striped = random.sample((0, self.cube_dim[1]), num_stripes)

        return cols_striped