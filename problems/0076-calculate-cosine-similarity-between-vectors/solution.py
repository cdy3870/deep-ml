import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	dot = v1 @ v2.T

	magnitude_product = np.sqrt(np.square(v1).sum()) * np.sqrt(np.square(v2).sum())

	return dot / magnitude_product