
import numpy as np

def r_squared(y_true, y_pred):
	SSR = np.square(y_pred - y_true).sum()
	SST = np.square(y_pred - y_true.mean()).sum()
	divided = SSR/SST


	if SST == 0:
		divided = 1
		

	return 1 - divided