import os
import sys
from PIL import Image
import numpy as np
import pca
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_images(rootdir):
	im_matrix = []
	for (dirpath, dirnames, filenames) in os.walk(rootdir):
		for dirname in dirnames:
			subj_path = os.path.join(dirpath, dirname)
			for f in os.listdir(subj_path):
				try:
					im = Image.open(os.path.join(subj_path, f))
					im_matrix.append(np.asarray(im.getdata()))
				except:
					pass
	return im_matrix

im_matrix = np.array(read_images('faces_data/'), dtype='float64')

evals, evecs, mean_X = pca.PCA(im_matrix)
evecs = np.asarray(evecs)

e_plot = evals / max(evals)
plt.hist(e_plot)
plt.ylabel('Singular values')
plt.savefig('sing_eigval.png')
plt.clf()

tot = sum(evals)
var_exp = [(i / tot)*100 for i in evals]
cum_var_exp = np.cumsum(var_exp)
x_graph = range(len(evals))
plt.plot(x_graph, cum_var_exp)
plt.xlabel('Number of eigen-values')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('continuous_m.png')
plt.clf()
print('Number of PCs required to explain:')
tmp = np.where(cum_var_exp >= 50)[0][0]
print('50-percent variance = %s' % tmp)
tmp = np.where(cum_var_exp >= 80)[0][0]
print('80-percent variance = %s' % tmp)
tmp = np.where(cum_var_exp >= 90)[0][0]
print('90-percent variance = %s' % tmp)

plt.figure()
plt.suptitle('Mean Face')
plt.imshow(mean_X.reshape(30,32),cmap='Greys_r')
plt.savefig('meanface.png')

for i in range(0,10):
	plt.figure()
	plt.suptitle('Eigen Faces')
	plt.imshow(evecs[i,:].reshape(30,32), cmap='Greys_r')
	plt.savefig('eigenface' + str(i) + '.png')

num_comp = [1,2,10,100]
orig_face = im_matrix[0]
plt.figure()
plt.suptitle('Original Face')
plt.imshow(orig_face.reshape(30, 32),cmap='Greys_r')
plt.savefig('origface.png')

std_orig_face = im_matrix[0] - mean_X

for num in num_comp:
	projected = np.dot(evecs[0:num, :], std_orig_face)
	reconstruct = np.dot(evecs[0:num, :].T, projected) + mean_X
	plt.figure()
	plt.suptitle('Reconstructed Image with ' + str(num) + ' components')
	plt.imshow(reconstruct.reshape(30,32),cmap='Greys_r')
	plt.savefig('reconst_' + str(num) + '_comp.png')
