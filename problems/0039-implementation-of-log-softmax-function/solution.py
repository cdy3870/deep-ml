import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	s = np.array(scores)
	return np.log(np.exp(s) / np.exp(s).sum())