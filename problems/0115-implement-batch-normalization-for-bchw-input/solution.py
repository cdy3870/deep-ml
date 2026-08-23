import numpy as np
def batch_normalization(X, gamma, beta, running_mean=None, running_var=None,
                        momentum=0.1, epsilon=1e-5, training=True):
    B, C, H, W = X.shape

    if running_mean is None:
        running_mean = np.zeros((1, C, 1, 1))
    if running_var is None:
        running_var = np.ones((1, C, 1, 1))

    if training:
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        variance = np.square(X - mean).mean(axis=(0, 2, 3), keepdims=True)

        normalized = (X - mean) / np.sqrt(variance + epsilon)

        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * variance
    else:
        normalized = (X - running_mean) / np.sqrt(running_var + epsilon)

    scale_and_shifted = gamma * normalized + beta
    return scale_and_shifted, running_mean, running_var