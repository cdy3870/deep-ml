import numpy as np

def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    if "num_kv_heads" not in model_config:
        num_kv_heads = model_config["num_attention_heads"]
    else:
        num_kv_heads = model_config["num_kv_heads"]

    total_elements = batch_size * model_config["num_layers"] * 2 * num_kv_heads * (model_config["hidden_size"] / model_config["num_attention_heads"]) * seq_len

    bytes = total_elements * model_config["dtype_bytes"]

    megabytes = np.round(bytes / (1024 * 1024), 4)

    megabytes_per_layer = np.round(megabytes / model_config["num_layers"], 4)

    memory_per_token =  model_config["num_layers"] * 2 * num_kv_heads * (model_config["hidden_size"] / model_config["num_attention_heads"]) * model_config["dtype_bytes"]

    rounded = np.round(memory_per_token / 1024, 4)

    return {"kv_cache_elements": total_elements,
            "kv_cache_size_bytes": bytes,
            "kv_cache_size_mb": megabytes,
            "per_layer_size_mb": megabytes_per_layer,
            "per_token_size_kb": rounded}