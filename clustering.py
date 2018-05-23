from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
from time import time
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()
def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)
# #############################################################################
from nltk.corpus import stopwords
print ("\nESET accessing datasets with scores and dissimilarities...")
# Open and read a bunch of files
f = open('althesim.txt')
doc1 = str(f.read())
f = open('sci.crypt.txt')
doc2 = str(f.read())
f = open('talk.politics.guns.txt')
doc3 = str(f.read())
f = open('comp.windows.x2.txt')
doc4 = str(f.read())
f = open('compgraph.txt',)
doc5 = str(f.read())
f = open('rec.sport.baseball.txt')
doc6 = str(f.read())
f = open('rec.autos2.txt')
doc7 = str(f.read())
f = open('comp.os.ms-windows.misc2.txt')
doc8 = str(f.read())
f = open('talk.religion.misc.txt')
doc9 = str(f.read())
f = open('rec.sport.hockey.txt')
doc10 = str(f.read())
f = open('comp.sys.mac.hardware2.txt')
doc11 = str(f.read())
f = open('sci.electronics.txt')
doc12 = str(f.read())
f = open('soc.religion.christian.txt')
doc13 = str(f.read())
f = open('misc.forsale2.txt')
doc14 = str(f.read())
f = open('sci.med.txt')
doc15 = str(f.read())
f = open('compsysibm.txt')
doc16 = str(f.read())
f = open('sci.space.txt')
doc17 = str(f.read())
f = open('rec.motorcycles.txt')
doc18 = str(f.read())
f = open('talk.religion.misc.txt')
doc19 = str(f.read())

f = open('talk.politics.mideast.txt')
doc20 = str(f.read())
doc21= 'Religion, God and People'
doc22='Politics,Mideast, Turkish'
doc23='File,windows,graphics'
doc24='electronics,encryption,space'
doc25='motorcycles,dod, bikes'

example =[ doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17, doc18, doc19, doc20, doc21]
target=[doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17,doc19]

Cardinality=0
for files in example:
    if files.endswith('.txt'):
        Cardinality+=1
vectorizer = CountVectorizer(input='example')
dtm = vectorizer.fit_transform(example)  # a sparse matrix
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
import matplotlib.pyplot as plt


ind = np.arange(N)  # points on the x-axis

width = 0.5

plt.bar(ind, dtm[:,0], width=width)
plt.xticks(ind + width/2, dtm)  # put labels in the center
plt.title('Share of Topic #0')
plt.show()
dist = euclidean_distances(dtm)
print (np.round(dist, 1))

dist = 1-cosine_similarity(dtm)
print (np.round(dist, 2))
norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))  # multiplication between arrays is element-wise
dtm_normed = dtm / norms
similarities = np.dot(dtm_normed, dtm_normed.T)

print ("special coding", dist[1,19], dist[2,19],dist[3,19],dist[4,19],dist[5,19],dist[6,19],dist[7,19],dist[8,19], dist[9,19],dist[10,19], dist[11,19], dist[12,19],dist[13,19],dist[14,19],dist[15,19],dist[16,19],dist[17,19],dist[18,19], dist[19,19],dist[20,19], )
#print (dist[20,19])
print ("special coding2",dist[1,20], dist[2,20],dist[3,20],dist[4,20],dist[5,20],dist[6,20],dist[7,20],dist[8,20], dist[9,20], dist[10,20],dist[11,20], dist[12,20], dist[13,20], dist[14,20], dist[15,20], dist[16,20], dist[17,20],dist[18,20], dist[19,20],dist[20,20])
print ("special coding3",dist[1,5], dist[2,5],dist[3,5],dist[4,5],dist[5,5],dist[6,5],dist[7,5],dist[8,5], dist[9,5], dist[10,5],dist[11,5], dist[12,5], dist[13,5], dist[14,5], dist[15,5], dist[16,5], dist[17,5],dist[18,5], dist[19,5],dist[20,5])
print ("special coding4",dist[1,17], dist[2,17],dist[3,17],dist[4,17],dist[5,17],dist[6,17],dist[7,17],dist[8,17], dist[9,17], dist[10,17],dist[11,17], dist[12,17], dist[13,17], dist[14,17], dist[15,17], dist[16,17], dist[17,17],dist[18,17], dist[19,17],dist[20,17])
print ("special coding5",dist[1,18], dist[2,18],dist[3,18],dist[4,18],dist[5,18],dist[6,18],dist[7,18],dist[8,18], dist[9,18], dist[10,18],dist[11,18], dist[12,18], dist[13,18], dist[14,18], dist[15,18], dist[16,18], dist[17,18],dist[18,18], dist[19,18],dist[20,18])


from scipy.spatial import KDTree
r=(np.round(dist, 1)).T

print ("\n EUCLIDEAN DISSIMILARITY of .txt files in: ")
print (np.round(similarities, 1))

vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
vec=vectorizer.fit(example)   # train vec using list1
vectorized = vec.transform(example)   # transform list1 using vec
km=KMeans(n_clusters=10, init='k-means++', n_init=1, max_iter=1000, verbose=0, random_state=None, n_jobs=1)
km.fit(vectorized)
list2Vec=vec.transform(target)  # transform list2 using vec

print (list2Vec)
targets=km.predict(list2Vec)
targets2=km.predict(vectorized)
print (targets2)
print(targets)

Cardinality=0
for files in example:
    if files.endswith('.txt'):
        Cardinality+=1
dtm = vectorizer.fit_transform(example)
#print(np.unique(dtm))

#type(dtm)
pd.DataFrame(dtm.toarray(),index=example,columns=vectorizer.get_feature_names ()).head(8)
vocab=vectorizer.get_feature_names()
frame= dtm.toarray()  # convert to a regular array
#print (dtm.shape)
N, K = frame.shape
ind = np.arange(N)  # points on the x-axis
width = 0.2
vocab = np.array(vocab)
n, _ = frame.shape
dist = np.zeros((n, n))
#dissimilarity

lsa = TruncatedSVD(2, algorithm = 'arpack')

cv = CountVectorizer(input='example',strip_accents='ascii')
dtMatrix = cv.fit_transform(example).toarray()
print (dtMatrix.shape)
featurenames = cv.get_feature_names()
print (featurenames)
#Tf-idf Transformation
tfidf = TfidfTransformer()
tfidfMatrix = tfidf.fit_transform(dtMatrix).toarray()
print (tfidfMatrix.shape)
n_topics = 15
svd = TruncatedSVD(n_components =n_topics)
svdMatrix = svd.fit_transform(tfidfMatrix)

print (svdMatrix)

#Cosine-Similarity
#cosine = cosine_similarity(svdMatrix[1], svdMatrix)

dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
ll=pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = vectorizer.get_feature_names())
print(ll)


#n_clusters = which defines the k numbers of clusters you want.
#n_init = Number of times k-means will run with different seeds. This is helpful since k-means can lead to local minimum.
#init: Initialization strategy. Few possible options are ‘k-means++’, ‘random’. K-means++ selects the k cluster is a smarter way to speed up convergence, whereas random selects at random k samples from the dataset.
#max_iter: Total number of times the k-means algorithm will run maximum number of times for a given run.
#After convergence, the k-means algorithm returns the following parameters:
#labels_: labels of each input sample
#interia_: The final value which we were minimizing.
#cluster_centers_ [k, n_features]: An array consisting of the coordinates of the centroid for each cluster. The dimension of the centroid (features) is same as the dimension of the each of the input sample(d-dimension).

NUM_TOPICS = 15
true_k = 15
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
x=model.fit(dtm_lsa)
print (x)

print("Top terms per cluster:")
n_topics = 15
y=pd.DataFrame(dtm_lsa, index = example, columns = ["component_1","component_2" ])
print(y)
xs = [w[0] for w in dtm_lsa]
ys = [w[1] for w in dtm_lsa]
xs, ys
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xs,ys)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Plot of points against LSA principal components')
plt.show()
similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
np.asmatrix(dtm_lsa)
print(np.asmatrix(dtm_lsa).T)

xx=pd.DataFrame(similarity,index=example, columns=example).head(8)
print(similarity)
print(np.mean(similarity))

# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1, algorithm = 'arpack',
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(dtm)
clusters = km.labels_.tolist()

# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(dtm)
print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in different topic spaces
print(lsi_Z[0])
def print_topics(model, vectorizer, top_n=8):
    for idx, topic in enumerate(model.components_):
        print("Concepts %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

labels = targets2

#print(labels)
true_k =  np.unique(labels)

print("LSI Model:")
print_topics(lsi_model, vectorizer)
print("=" * 20)

print("done in %0.3fs" % (time() - t0))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))


print()

