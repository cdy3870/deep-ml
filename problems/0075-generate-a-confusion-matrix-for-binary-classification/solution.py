
from collections import Counter
import numpy as np
def confusion_matrix(data):
	y_true = np.array(data)[:, 0]
	y_pred = np.array(data)[:, 1]

	TP = np.sum((y_true == 1) & (y_pred == 1))
	FN = np.sum((y_true == 1) & (y_pred == 0))
	FP = np.sum((y_true == 0) & (y_pred == 1))
	TN = np.sum((y_true == 0) & (y_pred == 0))

	return [[int(TP), FN], [FP, TN]]
