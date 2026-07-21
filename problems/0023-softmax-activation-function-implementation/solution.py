import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_e = max(scores)

    exp = [math.exp(s - max_e) for s in scores]
    total = sum(exp)

    softmax_activation = [(e)/total for e in exp]

    return softmax_activation