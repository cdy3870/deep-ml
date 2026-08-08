import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	trace = matrix[0][0] + matrix[1][1]
	det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

	quadratic_1 = (trace + math.sqrt(trace ** 2 - (4 * det)))/2

	quadratic_2 = (trace - math.sqrt(trace ** 2 - (4 * det)))/2

	return [quadratic_1, quadratic_2]