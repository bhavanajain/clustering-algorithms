import numpy as np
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

# Input data into X
X = np.genfromtxt('dataset1.txt', delimiter=" ", dtype=float)

# Plot the graph of number of clusters vs Inertia - Sum of distances of samples to their closest cluster center
def plot_inertia(X, max_k = 20):
	x_graph = range(1, max_k)
	y_graph = []
	for k in range(1, max_k):
		kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
		y = kmeans.labels_
		curr_inertia = kmeans.inertia_
		y_graph.append(curr_inertia)
	plt.plot(x_graph, y_graph)
	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia')
	plt.savefig('1a_inertia.png')
	plt.clf()
	return y_graph

inertia_arr = plot_inertia(X)

# Elbow of the graph gives us the best k = 6
best_k = 6	
kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X)
y = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the points and their centroids.
colors = plt.cm.spectral(y.astype(float)/kmeans.n_clusters)
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.7, s=30, lw=0,c=colors)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=30, linewidths=2, color='k')
plt.savefig('1a_kmeans.png')
