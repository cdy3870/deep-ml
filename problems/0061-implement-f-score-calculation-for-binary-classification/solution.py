import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""

	tp = np.where((y_pred == 1) & (y_true == 1), 1, 0).sum()
	fp = np.where((y_pred == 1) & (y_true == 0), 1, 0).sum()

	precision =  tp / (fp + tp)

	fn = np.where((y_pred == 0) & (y_true == 1), 1, 0).sum()

	recall = tp / (fn + tp)
	
	f1 = (1 + np.square(beta)) * (precision * recall)/((np.square(beta) * precision) + recall)

	return np.round(f1, 4)