import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	"""
	Compute temperature-scaled softmax probabilities from logits.
	
	Args:
		logits: 1D numpy array of raw model output scores
		temperature: float controlling distribution sharpness
	
	Returns:
		List of probabilities after temperature scaling
	"""
	# Your code here

	if temperature == 0:
		temp_array = np.zeros((len(logits), ))
		index = np.argmax(logits)
		temp_array[index] = 1
		return temp_array
	# print(np.zeros((len(logits), )))

	maxes = np.max(logits, axis=-1, keepdims=True)
	probs = np.exp((logits - maxes) / temperature) / np.exp((logits - maxes) / temperature).sum()

	return probs
