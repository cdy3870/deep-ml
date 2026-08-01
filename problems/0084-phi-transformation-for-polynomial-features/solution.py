import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# Your code here
	
	new_list = []
	for d in data:
		temp_list = []
		for i in range(degree + 1):
			temp_list.append(d ** i)

		new_list.append(temp_list)

	return new_list
