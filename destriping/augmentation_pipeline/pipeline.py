import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to the dataset", type=str)
parser.add_argument("add_stripes", help="should we add stripes", type=bool)
parser.add_argument(
    "num_stripes", help="stripes per image, 0 for randomly chosen in bounds", type=int
)
parser.add_argument(
    "data_save", help="path where the dataset should be saved", type=str
)
args = parser.parse_args()


if add_stripes:
    data = np.load(args.data).astype(np.float16)
    augmented = add_multiplicative_stripes(data, args.num_stripes)

# def plot_band(dataset, band_no=90):
#     plt.figure(figsize=(8, 6))
#     plt.imshow(dataset[:,:, band_no], cmap='jet')
#     plt.title(f'Band-{band_no}', fontsize=14)
#     plt.axis('off')
#     plt.colorbar()
#     plt.show()


def add_multiplicative_stripes(data_cube, stripe_per_image=0):
    """Add multiplicative stripes to original data set which has no stripes.
    Stripes are added randomly and have no specific patterns.

    Args:
    data_cube: orginal data cube, where dimention is of form [spatial X spatial X spectral]
    stripe_per_image: optional variable, if number is given, the code will generate that amount of stripes on each frame. If number not given, the code will randomly choose the number of stripes for each frame

    Returns:
    striped_data: data cube with stripes added to it

    """
    data_cube = data_cube.astype(np.float16)
    cube_dim = data_cube.shape  # cube dimesion
    striped_data = copy.deepcopy(
        data_cube
    )  # copies original data to avoid changing original data

    # random number of stripes for the frame, 25% to 60% of col are striped
    if stripe_per_image == 0:
        # choosing how many stripes for each frame
        num_stripes = np.random.randint(
            low=int(0.25 * cube_dim[1]), high=int(0.6 * cube_dim[1]), size=cube_dim[2]
        )
    else:
        num_stripes = np.full(
            shape=cube_dim[2], fill_value=stripe_per_image, dtype=np.int
        )

    for i in range(0, cube_dim[2], 1):  # going through each frame
        # create a list of non repeating int numbers for size of data cube, choosing which columns will be striped
        col_stripes = random.sample(range(0, cube_dim[1]), num_stripes[i])

        # create list of repeating random number, multiplier is choosen b/w 0.5 to 1.6
        multiplier = np.random.randint(low=5, high=16, size=num_stripes[i]) / 10

        # go through each column that we will add stripes and mutiply the column values by the multiplier
        for k in range(0, len(col_stripes), 1):
            striped_data[:, col_stripes[k], i] *= multiplier[k]

    return striped_data


np.save(augmented, args.data_save)
