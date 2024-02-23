Move all the hyperspectral images to test with into the GRN-HSI-Denoising-TO-TEST/data/datasets/ICVL_test_pair_whole_image/stage1/icvl_50/ folder
"run readfns.m" to read the filenames of the test datasets.

"run demo_add_iidgaussian.m" to format the test data into the format required for the model. We are not using their noise. The code uses the apply_stripes function to add noise over the clean data. 

"run test.py" to test the GRN network.
