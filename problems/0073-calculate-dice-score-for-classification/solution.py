
import numpy as np

def dice_score(y_true, y_pred):
	# Write your code here

	TP = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
	
	nume = 2 * TP
	FP = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
	FN = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
	denom = 2 * TP + FP + FN

	if denom == 0:
		return 0

	return nume/denom
