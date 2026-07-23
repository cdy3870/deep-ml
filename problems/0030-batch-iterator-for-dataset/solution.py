import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	batches = []
	start = 0
	if y is not None:
		for i in range(int(len(X) / batch_size) + 1):
			batches.append([X[start:start+batch_size], y[start:start+batch_size]])
			start += batch_size
	else:
		for i in range(int(len(X) / batch_size) + 1):
			batches.append([X[start:start+batch_size]])
			start += batch_size
	return batches