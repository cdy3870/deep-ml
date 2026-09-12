import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    Calculate METEOR score for machine translation evaluation.
    
    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)
    
    Returns:
        METEOR score between 0 and 1
    """
    # Your code here
    
    def tokenize(sentence):
        return sentence.lower().split(" ")

    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    ref_tokens_copy = ref_tokens.copy()

    unigram_matches = 0
    for token in cand_tokens:
        if token in ref_tokens_copy:
            unigram_matches += 1
            ref_tokens_copy.remove(token)

    # print(unigram_matches)


    prec = unigram_matches / len(cand_tokens)
    rec = unigram_matches / len(ref_tokens)
    f_mean = (prec * rec) / (alpha * prec + (1 - alpha) * rec)
    
    def count_chunks():
        ref_pool = list(enumerate(ref_tokens))  # (index, token) pairs, consumable
        matched_ref_indices = []
        for token in cand_tokens:
            for j, (idx, ref_tok) in enumerate(ref_pool):
                if ref_tok == token:
                    matched_ref_indices.append(idx)
                    ref_pool.pop(j)
                    break

        num_chunks = 0
        prev_idx = None
        for idx in matched_ref_indices:
            if prev_idx is None or idx != prev_idx + 1:
                num_chunks += 1
            prev_idx = idx
        return num_chunks
    

    penalty = gamma * (count_chunks() / unigram_matches) ** beta

    return f_mean * (1 - penalty)

    # return 1



