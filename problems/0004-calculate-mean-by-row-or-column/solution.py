import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == "column":
		means = np.mean(np.array(matrix), axis=0)
	else:
		means = np.mean(np.array(matrix), axis=1)

	return means