import numpy as np
import math

# compute the euclidean distance between two vectors
def euclid_distance(a, b):
		return math.sqrt(np.power(a-b,2).sum())

class DBscan(object):

	def __init__(self, eps = 0.5, min_samples=5):
		self.eps=eps
		self.min_samples = min_samples
		self.UNCLASSIFIED = -1
		self.NOISE = 0

	def regionQuery(self, X, pt_index):
		"""
		Returns the eps-neighbourhood of X[pt_index, :]
		"""
		seeds = []
		n_points = X.shape[0]
		for i in range(0, n_points):
			if euclid_distance(X[i, :], X[pt_index, :]) < self.eps:
				seeds.append(i)
		return seeds

	def expandCluster(self, X, index, y, clusterId):
		"""
		Steps:
		1. Check the eps-neighbourhood of X[index]
			a. if it contains lesser points than min_samples - label as noise
			b. else label all seeds (will include index) - clusterId
				i. Now look at the eps-neighbourhood of all seeds - result
				ii. If it contains an unclassified sample, label it clusterId and 
					add it to seeds to examine its eps-neighbourhood.
				iii. Pop the examined point.

		"""
		seeds = self.regionQuery(X, index)
		if len(seeds) < self.min_samples:
			y[index] = self.NOISE
			return False
		else:
			for i in seeds:
				y[i] = clusterId
			seeds.remove(index)
			while(len(seeds) > 0):
				curr = seeds[0]
				result = self.regionQuery(X, curr)
				if (len(result) >= self.min_samples):
					for i in result:
						if y[i] == self.UNCLASSIFIED or y[i] == self.NOISE:
							if y[i] == self.UNCLASSIFIED:
								seeds.append(i)
							y[i] = clusterId
				seeds = seeds[1:]
		return True

	def run(self, X):

		# mark all samples as unclassified
		n_points = X.shape[0]
		y = [self.UNCLASSIFIED] * n_points;

		cluster_count = self.NOISE + 1
		for i in range(0, n_points):
			curr = X[i, :]
			if y[i] == self.UNCLASSIFIED:
				if self.expandCluster(X, i, y, cluster_count):
					cluster_count += 1
		return y