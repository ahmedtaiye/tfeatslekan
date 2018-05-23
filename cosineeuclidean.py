from __future__ import print_function
import sklearn
import mpl_toolkits
import os  # for os.path.basename
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
sklearn.__version__
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

filenames = ['ken-lay_body.txt',
             'jeff-skilling_body.txt',
             'Richard-shapiro_body.txt',
             'kay-mann_body.txt',
             'Jeff-dasovich_body.txt',
             'tana jones_body.txt',
             'steven kean_body.txt',
             'shackleton sara_body.txt',
             'james steffes_body.txt',
             'Mark taylor_body.txt',
             'davis pete_body.txt',
             'Chris g_body.txt',
             'kate symes_body.txt']
#Calculating the cardinality and dissimilarities.
Cardinality=0
for files in filenames:
    if files.endswith('.txt'):
        Cardinality+=1
vectorizer = CountVectorizer(input='filename')
dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
vocab = vectorizer.get_feature_names()  # a list
#type(dtm)
dtm = dtm.toarray()  # convert to a regular array
#print (dtm.shape)
N, K = dtm.shape
ind = np.arange(N)  # points on the x-axis
width = 0.2
vocab = np.array(vocab)
n, _ = dtm.shape
dist = np.zeros((n, n))

#dissimilarity
Dissimilarity=dist
for i in range(n):
     for j in range(n):
        x, y = dtm[i, :], dtm[j, :]
        dist[i, j] = np.sqrt(np.sum((x - y)**2))

#A sparse matrix only records non-zero entries and is used to store matrices that contain a significant number of entries
#  that are zero.To understand why this matters enough that CountVectorizer returns a sparse matrix by default,
# consider a 4000 by 50000 matrix of word frequencies that is 60% zeros. In Python an integer takes up four bytes,
# so using a sparse matrix saves almost 500M of memory, which is a considerable amount of computer memory in the 2010s.
# (Recall that Python objects such as arrays are stored in memory, not on disk). If you are working with a very
# large collection of texts, you may encounter memory errors after issuing the commands above. Provided your corpus is not truly
# massive, it may be advisable to locate a machine with a greater amount of memory. For example, these days
# it is possible to rent a machine with 64G of memory by the hour. Conducting experiments on a random subsample
# (small enough to fit into memory)
#is also recommended.
plt.bar(ind, dtm[:,0], width=width)
plt.xticks(ind + width, filenames)  # put labels in the center

plt.title('Share of Topic #0')
dist = euclidean_distances(dtm)
print (np.round(dist, 1))

#Keep in mind that cosine similarity is a measure of similarity (rather than distance) that ranges between 0 and 1 (as it is the cosine of the angle between the two vectors).
# In order to get a measure of distance (or dissimilarity), we need to “flip” the measure so that a larger angle receives a larger value. The distance measure
#  derived from cosine similarity is therefore one minus the cosine similarity between two vectors.
dist = 1 - cosine_similarity(dtm)
print (np.round(dist, 2))
norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))  # multiplication between arrays is element-wise
dtm_normed = dtm / norms
similarities = np.dot(dtm_normed, dtm_normed.T)


print ("\n COSINE DISSIMILARITY of .txt files in: ")
print (np.round(similarities, 2))

from scipy.spatial import KDTree
r=(np.round(dist, 1)).T

print ("\n EUCLIDEAN DISSIMILARITY of .txt files in: ")
print (np.round(similarities, 1))


mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
# short versions of filenames:
# convert 'data/austen-brontë/Austen_Emma.txt' to 'Austen_Emma'
names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]
# color-blind-friendly palette
for x, y, name in zip(xs, ys, names):
     color = 'red' if "ken-lay_body" in name else 'skyblue'
     plt.scatter(x, y, c=color)
     plt.text(x, y, name)
plt.show()
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
     ax.text(x, y, z, s)
plt.show()
linkage_matrix = ward(dist)
# match dendrogram to that returned by R's hclust()
dendrogram(linkage_matrix, orientation="right", labels=names)
plt.tight_layout()  # fixes margins
plt.show()



