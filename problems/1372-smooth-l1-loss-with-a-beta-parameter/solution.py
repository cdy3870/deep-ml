import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    # Your code here

    if beta <= 0:
        raise(ValueError())
    
    x = np.array(pred) - np.array(target)

    loss = np.where(x < beta, (0.5 * x ** 2) / beta, abs(x) - 0.5 * beta)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss

    raise(ValueError())

