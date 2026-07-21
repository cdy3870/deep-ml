import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_array = np.array(X)

	# print(X_array.T @ X_array)
	theta = np.linalg.inv(X_array.T @ X_array) @ (X_array.T @ y)
	return theta