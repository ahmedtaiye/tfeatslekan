
from __future__ import print_function
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.cluster import KMeans
import sklearn.cluster.k_means_
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import ProjectedGradientNMF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#import codecs, difflib, Levenshtein, distance
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
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

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')
# Bring in standard stopwords
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
from nltk.corpus import stopwords
data = []
# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = """To  [ ] I you am As it  can't  <<...>>  sincerely, .  > - < Kenneth Lay/Corp/Enron@Enron Best regards Sincerely From  Sent Original Message Q <-> * | /\ 100% 12345678910 () """

# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords.split()

stopWords = stopwords.words('english')
print ("\nCalculating document similarity scores...")
# Open and read a bunch of files
f = open('ken-lay_body.txt')
doc1 = str(f.read())
f = open('jeff-skilling_body.txt')
doc2 = str(f.read())
f = open('Richard-shapiro_body.txt')
doc3 = str(f.read())
f = open('kay-mann_body.txt')
doc4 = str(f.read())
f = open('Jeff-dasovich_body.txt',)
doc5 = str(f.read())
f = open('tana jones_body.txt')
doc6 = str(f.read())
f = open('steven kean_body.txt')
doc7 = str(f.read())
f = open('shackleton sara_body.txt')
doc8 = str(f.read())
f = open('james steffes_body.txt')
doc9 = str(f.read())
f = open('Mark taylor_body.txt')
doc10 = str(f.read())
f = open('davis pete_body.txt')
doc11 = str(f.read())
f = open('Chris g_body.txt')
doc12 = str(f.read())
f = open('kate symes_body.txt')
doc13 = str(f.read())
f = open('Mcconnell.body.txt')
doc14 = str(f.read())
f = open('kaminski_body.txt')
doc15 = str(f.read())
#train_string = 'By these proceedings for judicial review the Claimant seeks to challenge the decision of the Defendant dated the 23rd of May 2014 refusing the Claimant’s application of the 3rd of January 2012 for naturalisation as a British citizen'
# Construct the training set as a list
train_set = [ doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15]
# Set up the vectoriser, passing in the stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)
# Apply the vectoriser to the training set
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)
c = cosine_similarity(tfidf_matrix_train)

D = euclidean_distances(tfidf_matrix_train)

#print ("\nSimilarity Score [*] ",jaccard_similarity_score(tfidf_matrix_train[0:1], tfidf_matrix_train)
from sklearn.metrics.pairwise import manhattan_distances
r= manhattan_distances(tfidf_matrix_train[0:1], tfidf_matrix_train)
E=1-(r)/100
#print ("\nManhattan_distancesSimilarity Score [*] ",1-(manhattan_distances(tfidf_matrix_train[0:1], tfidf_matrix_train))/100)

from sklearn.metrics.pairwise import pairwise_distances
F= 1 - pairwise_distances(tfidf_matrix_train)
#print ("\nCosine_distancesSimilarity Score [*] ",cosine_distances(tfidf_matrix_train[0:1], tfidf_matrix_train))


NUM_TOPICS = 10
true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
x=model.fit(tfidf_matrix_train)
labels = x.labels_


#lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=10, learning_method='online')


print(c.shape)  # (NO_DOCUMENTS, NO_TOPICS)


# Let's see how the first document in the corpus looks like in different topic spaces
print(c[0])
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print("cosine similarity:")
#c_model=c(n_components=NUM_TOPICS)
#print_topics(c_model, tfidf_matrix_train)
print("=" * 20)
# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=10000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(tfidf_matrix_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(tfidf_matrix_train, km.labels_, sample_size=1000))

print()

print(c.shape)  # (NO_DOCUMENTS, NO_TOPICS)


# Let's see how the first document in the corpus looks like in different topic spaces
print(c[0])
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print("Euclidean:")
print("=" * 20)
# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=10000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(tfidf_matrix_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(tfidf_matrix_train, km.labels_, sample_size=1000))

print()
print(D.shape)  # (NO_DOCUMENTS, NO_TOPICS)


# Let's see how the first document in the corpus looks like in different topic spaces
print(D[0])
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


print("Manhanttan:")
print("=" * 20)
# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=10000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(tfidf_matrix_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(tfidf_matrix_train, km.labels_, sample_size=1000))

print()
print(E.shape)  # (NO_DOCUMENTS, NO_TOPICS)


# Let's see how the first document in the corpus looks like in different topic spaces
print(E[0])
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])



print("Jaccard:")
print("=" * 20)
# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=10000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(tfidf_matrix_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(tfidf_matrix_train, km.labels_, sample_size=1000))

print()
print(F.shape)  # (NO_DOCUMENTS, NO_TOPICS)


# Let's see how the first document in the corpus looks like in different topic spaces
print(F[0])
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
