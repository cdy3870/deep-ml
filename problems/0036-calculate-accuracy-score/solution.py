import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here

	y_sum = y_true + y_pred
	# print(np.where((y_sum == 2) | (y_sum == 0), 1, 0))
	# result = np.where((y_true == 2) | (y_true == 0), 1, 0).sum() / y_true.shape[1]
	result = np.where((y_sum == 2) | (y_sum == 0), 1, 0).sum() / y_true.shape[0]
	return result