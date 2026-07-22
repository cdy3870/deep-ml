import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here

	# print(np.array(features).shape)
	# print(np.array(weights).reshape(1, -1).shape)

	y_hat = np.array(weights).reshape(1, -1) @ np.array(features).T + bias
	
	
	probabilities = 1 / (1 + np.exp(-y_hat))

	mse = np.mean(np.square(np.array(probabilities) - np.array(labels)))


	return np.round(probabilities, 4).tolist()[0], np.round(mse, 4)