import numpy as np
from typing import List, Tuple
import numpy as np
def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """

    values = [i for i in range(n_samples)]

    if shuffle:
        np.random.shuffle(values)

    remainder = n_samples % k

    folds = []
    increment = 0
    for i in range(k):
        test_size = n_samples // k
        train_size = n_samples - test_size
    
        if i == 0:
            test_size += remainder
            train_size -= remainder

        test_split = values[increment:increment+test_size]
        folds.append(tuple([[v for v in values if v not in test_split], test_split]))
        increment += test_size
    return folds


