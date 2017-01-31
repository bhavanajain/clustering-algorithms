import dbscan
import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt('dataset1.txt', delimiter=" ", dtype=float)
est = dbscan.DBscan()
y = est.run(X);

print(set(y))

colors = plt.cm.spectral(np.asarray(y).astype(float)/(len(set(y))))
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.7, s=30, lw=0,c=colors)
plt.savefig('1b_dbscan.png')