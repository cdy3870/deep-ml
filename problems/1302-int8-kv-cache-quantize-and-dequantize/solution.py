import numpy as np

def quantize_kv_int8(x):
    """
    Symmetric INT8 quantize along the last axis.

    Returns (q, scale, x_hat).
    """
    
    scale = np.max(np.abs(x), axis=-1, keepdims=True) / 127
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(x / scale), -127, 127)

    x_hat = q * scale

    return q.astype(np.int8), scale, x_hat.astype(np.float64)
