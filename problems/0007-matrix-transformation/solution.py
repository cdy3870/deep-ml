import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	if len(A) > 2:
		T_inverse = np.linalg.inv(T)
		transformed_matrix = T_inverse @ np.array(A) @ np.array(S)

		return transformed_matrix



	# Check invertible
	T_det = T[0][0] * T[1][1] - T[0][1] * T[1][0]
	S_det = S[0][0] * S[1][1] - S[0][1] * S[1][0]


	if T_det == 0 or S_det == 0:
		return -1

	T_temp = [[T[1][1], -T[0][1]], [-T[1][0], T[0][0]]]
	T_inverse = 1/(T_det) * np.array(T_temp)

	transformed_matrix = T_inverse @ np.array(A) @ np.array(S)

	return transformed_matrix