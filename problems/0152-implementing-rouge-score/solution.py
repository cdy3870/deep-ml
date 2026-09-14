# Implement your function below.

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # Your code here
    
    ref_tokens = reference.lower().split(" ")
    ref_tokens_copy = ref_tokens.copy()
    cand_tokens = candidate.lower().split(" ")

    count = 0
    for token in cand_tokens:
        if token in ref_tokens:
            count += 1
            ref_tokens.remove(token)

    prec = count / len(cand_tokens)
    rec = count / len(ref_tokens_copy)
    f1 = (2 * prec * rec) / (prec + rec)

    return {"precision": prec, "recall": rec, "f1": f1}