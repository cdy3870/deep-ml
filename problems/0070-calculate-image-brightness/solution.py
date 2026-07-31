import numpy as np
def calculate_brightness(img):
	if len(img) == 0:
		return -1

	initial_size = len(img[0])
	for row in img:
		if len(row) != initial_size:
			return -1

	if np.array(img).max() > 255 or np.array(img).min() < 0:
		return -1

	return np.array(img).flatten().mean()
	
