import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	
	z = X @ weights.reshape(-1, 1) + bias
	sig = 1 / (1 + np.exp(-z))

	result = np.where(sig >= 0.5, 1, 0)
	# print(sig)

	return result.reshape(1,-1)[0,:]