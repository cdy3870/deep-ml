import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    x_new = x_new.reshape(1, -1)
    q = x_new @ W_Q
    k = x_new @ W_K
    v = x_new @ W_V
    if cache is not None:
        new_K_cache = np.concatenate([cache[0], k], axis=0)
        new_V_cache = np.concatenate([cache[1], v], axis=0)
    else:
        new_K_cache = k
        new_V_cache = v
    # print(q.shape)
    # print(new_K_cache.shape)
    scores = (q @ new_K_cache.T) / np.sqrt(x_new.shape[-1])
    output = ((np.exp(scores - np.max(scores))) / (np.exp(scores - np.max(scores))).sum()) @ new_V_cache
    
    return output[0], (new_K_cache, new_V_cache)
    
