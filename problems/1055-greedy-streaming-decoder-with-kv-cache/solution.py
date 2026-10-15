import numpy as np

def greedy_stream(model, prompt_ids, max_new_tokens: int, eos_id: int) -> list:
    """
    Greedy streaming decoder backed by a KV cache.

    Args:
        model: object exposing prefill(prompt_ids) -> np.ndarray of logits
               and decode_step(token_id) -> np.ndarray of logits.
        prompt_ids: iterable of int token ids.
        max_new_tokens: maximum number of tokens to generate.
        eos_id: token id that terminates generation.

    Returns:
        List of generated token ids (excluding the prompt and EOS).
    """
    logits = model.prefill(prompt_ids)
    output = []
    count = 0
    while count < max_new_tokens:
        next_token = np.argmax(logits)
        if next_token == eos_id:
            break
        else:
            output.append(int(next_token))
            logits = model.decode_step(next_token)
            count += 1

    return output
