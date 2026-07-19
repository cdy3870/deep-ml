import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	X = np.array(vectors)

	# mean centering

	# print(X)
	# print(np.mean(X, axis=1))

	X_centered = X - np.mean(X, axis=1, keepdims=True)
	# print(X_centered)


	cov_mat = (1/(len(vectors[0]) - 1)) * (X_centered @ X_centered.T)
	
	# print(cov_mat)

	return cov_mat