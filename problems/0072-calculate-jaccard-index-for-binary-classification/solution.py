
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	y_true_a = np.array(y_true)
	y_pred_a = np.array(y_pred)

	intersection = np.where((y_true_a == 1) & (y_pred_a == 1), 1, 0)
	union = np.where((y_true_a == 1) | (y_pred_a == 1), 1, 0)

	result = intersection.sum() / union.sum()
	return round(result, 3)
