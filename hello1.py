from __future__ import print_function
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import completeness_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
import numpy as np
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

data =[ doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17, doc18, doc19,doc20]
target=[doc1, doc2, doc5, doc3, doc2, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17,doc19, doc18, doc20]
vectorizer = CountVectorizer()
n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data)
print("done in %0.3fs." % (time() - t0))
print()

vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
vec=vectorizer.fit(data)   # train vec using list1
vectorized = vec.transform(data)   # transform list1 using vec
km=KMeans(n_clusters=10, init='k-means++', n_init=1, max_iter=1000, verbose=0, random_state=None, n_jobs=1)

km.fit(vectorized)
list2Vec=vec.transform(target)  # transform list2 using vec

print (list2Vec)
targets=km.predict(list2Vec)

targets2=km.predict(vectorized)
print (targets2)
print(targets)

labels=targets

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation( max_iter=5, learning_method='online', learning_offset=50., random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(tf, km.labels_, sample_size=2000))

print()
