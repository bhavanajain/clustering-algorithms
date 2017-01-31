import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import normalize

def PCA(X):
	num_data, dim = X.shape
	# calculate mean face
	mean_X = X.mean(axis=0)
	# print(mean_X.shape)
	X = X - mean_X

	if dim > num_data:
		# print('here')
		C = np.dot(X, X.T)
		
		eig_vals, eig_vecs = np.linalg.eigh(C)
		true_eig_vecs = np.dot(X.T, eig_vecs)
		
		true_eig_vecs = normalize(true_eig_vecs, axis=0)
		eig_pairs = [(np.abs(eig_vals[i]), true_eig_vecs[:,i]) for i in range(len(eig_vals))]
		eig_pairs.sort(key=lambda x: x[0], reverse=True)

		evals, evecs = [], []
		for i in range(len(eig_vals)):
			evals.append(eig_pairs[i][0])
			evecs.append(eig_pairs[i][1])

	return evals, evecs, mean_X



