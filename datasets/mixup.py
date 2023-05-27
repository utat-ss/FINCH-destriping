ingort numpy as np
import tensorflow as tf

# Load .npy files
x = np.load('cuprite512.npy')
y = np.load('indian_pine_array.npy')

def mixup(x, y, alpha=0.2):
    """
    Performs mixup data augmentation.
    
    Arguments:
        x: A NumPy array representing cuprite512.
        y: A NumPy array representing indian_pine_array.npy.
        alpha: The alpha parameter for the beta distribution.
        
    Returns:
        x_mix: The mixed features.
        y_mix: The mixed labels.
    """
    
    # Computing mixup ratio from a beta distribution:
    mix_ratio = np.random.beta(alpha, alpha)
    
    # Creating a shuffled version of the data:
    x_shuffle = tf.random.shuffle(x)
    y_shuffle = tf.random.shuffle(y)
    
    # Mixing the original and shuffled data:
    x_mix = mix_ratio * x + (1 - mix_ratio) * x_shuffle
    y_mix = mix_ratio * y + (1 - mix_ratio) * y_shuffle

    return x_mix, y_mix

# Applying mixup
x_mix, y_mix = mixup(x, y)
