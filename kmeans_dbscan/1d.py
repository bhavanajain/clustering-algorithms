import numpy as np
from sklearn.cluster import KMeans
import dbscan
import matplotlib.pyplot as plt

# Input data into X
X = np.genfromtxt('dataset2.txt', delimiter=" ", dtype=float)

kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
y = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the points and their centroids.
colors = plt.cm.spectral(y.astype(float)/kmeans.n_clusters)
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.7, s=30, lw=0,c=colors)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=30, linewidths=2, color='k')
plt.savefig('1d_kmeans.png')
plt.clf()

est = dbscan.DBscan(eps=1.5)
y = est.run(X)
# print(set(y))
colors = plt.cm.spectral(np.asarray(y).astype(float)/len(set(y)))
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.7, s=30, lw=0,c=colors)
plt.savefig('1d_dbscan.png')
plt.clf()



