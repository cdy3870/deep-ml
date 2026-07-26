
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""

	counter = {}
	length = len(y)

	for i in y:
		if i in counter:
			counter[i] += 1
		else:
			counter[i] = 1

	divisors = []

	for k, v in counter.items():
		divisors.append(v/length)

	val = 1 - np.square(np.array(divisors)).sum()
	# print(val)
	# print(counter)
	# print(divisors)
	return round(val,3)