import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	max_val = x.max() + 1
	one_hot_encoding = []
	x_list = x.tolist()
	for i in range(len(x_list)):
		encoding = [0 for j in range(max_val)]
		encoding[x_list[i]] = 1
		one_hot_encoding.append(encoding)

	return one_hot_encoding