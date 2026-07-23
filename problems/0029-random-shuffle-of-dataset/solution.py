import numpy as np

def shuffle_data(X, y, seed=None):

	# rng = np.random.default_rng(seed=seed)

	# Your code here
	# combined_data = [[X[i], y[i]] for i in range(len(X))]
	# rng = np.random.default_rng(seed=seed)

	# rng.shuffle(combined_data)


	# X = np.array([combined_data[i][0] for i in range(len(X))])
	# y = np.array([combined_data[i][1] for i in range(len(y))])

	# # print(X)
	# # print(y)

	# return X, y

	# idx = rng.permutation(len(X))
	# return X[idx], y[idx]

	np.random.seed(seed)

	idx = np.random.permutation(len(X))
	return X[idx], y[idx]