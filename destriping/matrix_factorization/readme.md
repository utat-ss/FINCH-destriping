This is the code for the "Hyperspectral Image Denoising via Matrix Factorization and Deep Prior Regularization" paper.

The model is now built. In order to match the dimension of the output of this model, I added a Conv3D layer.

For now I'm using regular Adam optimizer, instead of the experimental one with modifications as specified in the paper, because .h5 file format doesn't store this optimizer. I'll try using "saved models" directory format of tensorflow to store the model information. I'll look into it and make the neccesary changes.

Because of the lack of advanced math background on my part, I'm not able to understand the custom loss function specified in the paper. So, I'm using binary crossentropy now.

I'm able to train the model, but the results are not good.
