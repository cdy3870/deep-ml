import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    logits = np.array(logits)
    probs = np.exp(logits) / np.exp(logits).sum()
    order = np.argsort(-probs, kind='stable')                # 1. descending order, keep indices
    sorted_probs = np.cumsum(probs[order])
    min_i = len(probs) - 1                         # 2. safe default if p is never reached
    for i in range(len(probs)):
        if sorted_probs[i] >= p:
            min_i = i
            break

    keep = order[:min_i + 1]                       # 3. map back to original positions
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return (filtered / filtered.sum()).tolist()