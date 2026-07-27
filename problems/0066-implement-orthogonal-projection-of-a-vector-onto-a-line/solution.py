import numpy as np
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	L = np.array(L)
	v = np.array(v)
	unit_vector_L = L @ L.T
	dot = v @ L.T
	fraction = dot/unit_vector_L
	# print(fraction.reshape(1, 1))

	# print(L.shape)
	return (fraction.reshape(-1, 1) @ L.reshape(-1, 1).T).tolist()[0]
	# print(dot @ L)
	# return v @ unit_vector_L
