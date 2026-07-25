import numpy as np
def precision(y_true, y_pred):
	
	tp = np.where((y_pred == 1) & (y_true == 1), 1, 0).sum()
	fp = np.where((y_pred == 1) & (y_true == 0), 1, 0).sum()
	# print(tp)
	# print(fp)
	return tp / (fp + tp)
