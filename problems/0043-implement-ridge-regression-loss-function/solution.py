import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	
	# print(X.shape)
	# print(w.shape)
	# print(y_true.shape)

	ridge_loss = np.square(y_true - X @ w.T).mean() + alpha * np.square(w).sum()

	return ridge_loss
