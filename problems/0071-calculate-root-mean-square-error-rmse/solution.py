
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	y_true_a = np.array(y_true)
	y_pred_a = np.array(y_pred)


	rmse_res = np.sqrt(np.square(y_true_a - y_pred_a).mean())



	return round(rmse_res,3)
