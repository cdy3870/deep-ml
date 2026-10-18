import numpy as np

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    # split then transpose
    B = Q.shape[0]
    H = num_heads
    T = Q.shape[1]
    # d = head_dim
    Hkv = num_kv_heads


    reshaped_Q = np.swapaxes(Q.reshape(B, T, H, -1), 1, 2)
    reshaped_K = np.swapaxes(K.reshape(B, T, Hkv, -1), 1, 2) 
    reshaped_V = np.swapaxes(V.reshape(B, T, Hkv, -1), 1, 2)

    G = H // num_kv_heads

    d = reshaped_Q.shape[-1]

    reshaped_K = np.repeat(reshaped_K, G, axis=1)

    reshaped_V = np.repeat(reshaped_V, G, axis=1)

    temp = reshaped_Q @ np.swapaxes(reshaped_K, -1, -2) / np.sqrt(d)
    temp = np.exp(temp - np.max(temp, keepdims=True, axis=-1))

    attention = (temp / temp.sum(keepdims=True, axis=-1)) @ reshaped_V

    attention = np.swapaxes(attention, 1, 2).reshape(B, T, H * d)

    return attention
    
