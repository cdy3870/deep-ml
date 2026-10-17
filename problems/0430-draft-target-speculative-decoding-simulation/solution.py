import numpy as np

def speculative_decode(draft_probs, target_probs, draft_tokens, random_coins):
    """
    Simulate one round of speculative decoding.
    
    Args:
        draft_probs: List of K numpy arrays, each a probability distribution over vocabulary
        target_probs: List of K+1 numpy arrays, each a probability distribution over vocabulary
        draft_tokens: List of K token indices proposed by the draft model
        random_coins: List of K floats in [0,1) for acceptance decisions
    
    Returns:
        Dictionary with keys:
            - 'accepted_tokens': list of int token indices
            - 'num_generated': int total tokens produced
            - 'acceptance_rate': float fraction of draft tokens accepted (rounded to 4 decimals)
    """
    accepted_tokens = []
    count = len(draft_tokens)
    accepted_tokens_count = 0
    for i in range(len(draft_probs)):
        token = draft_tokens[i]
        q = draft_probs[i][token]
        p = target_probs[i][token]



        acceptance_prob =  p / q
        acceptance_prob = min(1, acceptance_prob)
        accepted = True if random_coins[i] < acceptance_prob else False
        if not accepted:
            adjusted_dist = target_probs[i] - draft_probs[i]
            replacement_token_index = np.maximum(np.argmax(adjusted_dist), 0)
            accepted_tokens.append(replacement_token_index)
            break

        else:
            accepted_tokens_count += 1
            accepted_tokens.append(token)

    if accepted_tokens_count == len(draft_probs):
        accepted_tokens_count += 1
        accepted_tokens.append(np.argmax(target_probs[-1]))
        count += 1


    return {"accepted_tokens": accepted_tokens, "num_generated": len(accepted_tokens), "acceptance_rate": np.round(accepted_tokens_count / count, 4)}
