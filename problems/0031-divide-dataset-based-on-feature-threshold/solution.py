import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here
	filter_condition = X[:, feature_i] >= threshold
	other_condition = X[:, feature_i] < threshold
	first_subset = X[filter_condition]
	second_subset = X[other_condition]

	return [first_subset, second_subset]