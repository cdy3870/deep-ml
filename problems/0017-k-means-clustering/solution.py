import math 
import numpy as np
 
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here

	def euc_dist(point_1, point_2):
		sums = 0
		for i in range(len(point_1)):
			sums += (point_2[i] - point_1[i]) ** 2
		dist = math.sqrt(sums)

		return dist

	
	centroids_copy = initial_centroids.copy()
	for n in range(max_iterations):
		clusters = {}
		for point in points:
			closest_centroid_dist = 10000
			for i, centroid in enumerate(centroids_copy):
				dist = euc_dist(point, centroid)
				if dist < closest_centroid_dist:
					closest_centroid_dist = dist
					closest_cluster = i

			if closest_cluster not in clusters:
				clusters[closest_cluster] = [point]
			else:
				clusters[closest_cluster].append(point)
		# print(clusters)
		centroids_copy = []
		for v in range(len(clusters)):
			centroids_copy.append(tuple(np.array(np.mean(clusters[v], axis=0)).tolist()))

	return centroids_copy


