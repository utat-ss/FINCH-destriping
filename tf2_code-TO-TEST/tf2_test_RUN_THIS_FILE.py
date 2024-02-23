# stdlib
import os

# external
import scipy.io
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# stdlib
import copy

# pass
import random
import time

# external
# pass
# pass
# pass
import matplotlib.pyplot as plt
import numpy as np

# pass
import tf2_model as model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# pass

# from Indexes import sam, ergas
# import mat73


##################### Select GPU device ###################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###########################################################################

start = time.time()

model_path = "/Users/prithviseran/Documents/GitHub/FINCH-destriping/GRN-HSI-Denoising-TO-TEST/pretrained_model/ICVL_stage2"
# "/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage3/case4"
# "/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage1"
#'/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage2' <- best
# "/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage3/case1"
# "/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage3/case3"
# "/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage3/case2"
# "/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage1"
#'/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/pretrained_model/ICVL_stage3/case5'  # can be changed
data_path = "/Users/prithviseran/Documents/GitHub/FINCH-destriping/GRN-HSI-Denoising-TO-TEST/data/datasets/ICVL_test_pair_whole_image/stage1/icvl_50/"
result_path = "/Users/prithviseran/Documents/GitHub/FINCH-destriping/GRN-HSI-Denoising-TO-TEST/icvl_test/"


# model_path = './model/CAVE/'
# data_path = './data/CAVE_testing/cave_complex_case5/'
# result_path = './result/cave_complex5/'

height = 512
width = 512
channels = 31


def add_basic_stripes(data_cube, num_stripes=0):
    """Add stripes to original data set which has no stripes. Stripes are added
    randomly and have no specific patterns.

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


if __name__ == "__main__":
    # scipy.io.loadmat('test.mat')
    # img_tmp = scipy.io.loadmat("/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/test_fns.mat")

    # print(img_tmp)

    imgName = os.listdir(data_path)
    filename = os.listdir(data_path)
    filename_ori = os.listdir(data_path)
    for i in range(len(filename)):
        filename[i] = data_path + filename[i]

    num_img = len(filename)
    PSNR = np.zeros([num_img, channels])
    SSIM = np.zeros([num_img, channels])
    #   ERGAS = np.zeros([num_img])
    SAM = np.zeros([num_img])
    Name = []

    image = tf.placeholder(tf.float32, shape=(1, height, width, channels))
    final = model.Inference(image)
    output = tf.clip_by_value(final, 0.0, 1.0)
    #   output = tf.squeeze(out)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device("/gpu:0"):
            if tf.train.get_checkpoint_state(model_path):
                ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
                saver.restore(sess, ckpt)
                print("Loading model")

            for i in range(num_img):
                # data_dict = mat73.loadmat('data.mat')
                img_tmp = scipy.io.loadmat(filename[i])  # choose one dataset
                # print(img_tmp)
                # rain = img_tmp['input']
                # rain = add_basic_stripes(rain)
                label = img_tmp["gt"]
                rain = add_basic_stripes(label)
                rain = np.expand_dims(rain, axis=0)
                """indian_pines =
                np.load("/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-
                Denoising/indian_pine_array.npy")

                radiance_data = (indian_pines-1000)/500.

                new_indian_pines = []

                for j in range(len(radiance_data)):
                  new_indian_pines.append(np.tanh(normalize(cv2.resize(radiance_data[:, :, j], dsize=(height, width), interpolation=cv2.INTER_LINEAR))))
                  #indian_pines[:, :, i] = cv2.resize(indian_pines[0][:, :, i], dsize=(height, width), interpolation=cv2.INTER_LINEAR)

                new_indian_pines = np.array(new_indian_pines)

                new_indian_pines = np.moveaxis(new_indian_pines, 0, -1)
                new_indian_pines = np.expand_dims(new_indian_pines, axis=0)

                """

                derained = sess.run(output, feed_dict={image: rain})
                # new_indian_pines[:, :, :, 0:31]})
                # test_again = cv2.resize(indian_pines[:, :, 10], dsize=(height, width), interpolation=cv2.INTER_LINEAR)

                index = imgName[i].rfind(".")
                name = imgName[i][:index]
                Name.append(name)
                mat_name = name + "_denoised" + ".mat"
                denoised_result = {}
                denoised_result["derained"] = derained[0, :, :, :]
                denoised_result["rain"] = rain[0, :, :, :]

                fig, axes = plt.subplots(2, 2, figsize=(14, 7))
                axes[0, 0].pcolormesh(derained[0, :, :, 0], cmap="inferno")
                axes[0, 1].pcolormesh(rain[0, :, :, 0], cmap="inferno")
                axes[1, 1].pcolormesh(label[:, :, 0], cmap="inferno")
                # axes[1, 0].pcolormesh(new_indian_pines, cmap="inferno")

                plt.show()
                denoised_result["label"] = label
                scipy.io.savemat(os.path.join(result_path, mat_name), denoised_result)

                for c in range(channels):
                    psnr_ori = peak_signal_noise_ratio(label[:, :, c], rain[0, :, :, c])
                    PSNR[i, c] = peak_signal_noise_ratio(
                        label[:, :, c], derained[0, :, :, c]
                    )
                    # print(label[:,:,c])
                    SSIM[i, c] = structural_similarity(
                        label[:, :, c], derained[0, :, :, c], data_range=1.0
                    )

                #                  print(name + ': %d / %d images processed, psnr_ori=%f, psnr=%f, ssim=%f'
                #                        % (c+1,channels,psnr_ori, PSNR[i,c],SSIM[i,c]))
                print(
                    "The %2d /% d test dataset: %30s: MPSNR = %.4f, MSSIM = %.4f"
                    % (
                        i + 1,
                        num_img,
                        filename_ori[i],
                        np.mean(PSNR[i, :]),
                        np.mean(SSIM[i, :]),
                    )
                )
            #              ERGAS[i] = ergas(label,derained[0,:,:,:])
            # SAM[i] = sam(label,derained[0,:,:,:])
            Measure = {}
            Measure["PSNR"] = PSNR
            Measure["SSIM"] = SSIM
            #          Measure['ERGAS'] = ERGAS
            # Measure['SAM'] = SAM
            Measure["Name"] = Name
            scipy.io.savemat(os.path.join(result_path, "Measure.mat"), Measure)
            end = time.time()
            print("Testing is finished!")
            print(
                "Mean PSNR = %.4f, Mean SSIM = %.4f, Mean SAM= %.4f, Time = %.4f"
                % (np.mean(PSNR), np.mean(SSIM), np.mean(SAM), end - start)
            )
