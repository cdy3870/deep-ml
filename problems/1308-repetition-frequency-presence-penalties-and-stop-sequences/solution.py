import numpy as np

def apply_logits_processors(logits, generated_ids, presence_penalty,
                            frequency_penalty, repetition_penalty,
                            stop_sequences):
    """
    Apply repetition / frequency / presence penalties and detect stop suffixes.

    Returns (adjusted_logits_list, should_stop).
    """
    counts = {}
    count_ls = []
    logits = np.array(logits)
    for i in generated_ids:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1

    count_ls = [counts[i] if i in counts else 0 for i in range(len(logits)) ]

    
    seen = np.isin(np.arange(len(logits)), generated_ids)
    penalty = np.where(logits > 0, logits /repetition_penalty, logits * repetition_penalty)
    apply_rep = np.where(seen, penalty, logits)



    adjusted_logits = np.where((np.isin(np.arange(len(logits)), generated_ids)), apply_rep - presence_penalty, apply_rep)

    adjusted_logits -= frequency_penalty * np.array(count_ls)

    def matched_sequence(generated_ids, seq):
        start = -1
        for i in range(len(seq)):

            if seq[start] != generated_ids[start]:
                return False
            start -= 1

        return True

    should_stop = False
    for seq in stop_sequences:
        if len(seq) <= len(generated_ids) and len(seq) > 0:
            if matched_sequence(generated_ids, seq):
                should_stop = True


    return adjusted_logits, should_stop


    # return logits, False