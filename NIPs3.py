from __future__ import print_function
from nltk.corpus import wordnet as wn

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

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
f = open('file1.txt')
doc1 = str(f.read())
f = open('file2.txt')
doc2 = str(f.read())
f = open('file3.txt')
doc3 = str(f.read())
f = open('file4.txt')
doc4 = str(f.read())
f = open('file5.txt')
doc5 = str(f.read())
f = open('file6.txt')
doc6 = str(f.read())
f = open('file7.txt')
doc7 = str(f.read())
f = open('file8.txt')
doc8 = str(f.read())
f = open('file9.txt')
doc9 = str(f.read())
f = open('file10.txt')
doc10 = str(f.read())
f = open('file11.txt')
doc11 = str(f.read())


filenames = ['file1.txt', 'file2.txt','file3.txt',
             'file4.txt', 'file5.txt','file6.txt',
             'file7.txt','file8.txt', 'file9.txt',
             'file10.txt','file11.txt']
#filenames=[ doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17, doc18, doc19,doc20]
#target=[doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17,doc19]

vectorizer = CountVectorizer(input='filename')
dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
vocab = vectorizer.get_feature_names()  # a list
type(dtm)
dtm = dtm.toarray()  # convert to a regular array
vocab = np.array(vocab)
n, _ = dtm.shape

dist = np.zeros((n, n))

for i in range(n):
     for j in range(n):
         x, y = dtm[i, :], dtm[j, :]
         dist[i, j] = np.sqrt(np.sum((x - y)**2))


from sklearn.metrics.pairwise import euclidean_distances

dist = euclidean_distances(dtm)

np.round(dist, 1)
import os  # for os.path.basename

import matplotlib.pyplot as plt

from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]
for x, y, name in zip(xs, ys, names):
     color = 'orange' if "talk.politics.guns1" in name else 'red'
     plt.scatter(x, y, c=color)
     plt.text(x, y, name)

plt.show()

top_3= ['The', 'x', 'In', 'network', 'A', 'I', 'data', 'model', 'n', 'set', 'system', 'neural', 'used', 'one', 'output', 'networks', 'two', 'E', 'error', 'number', 'For', 'results', 'J', 'shown', 'parameters', 'L', 'w', 'O', 'models', 'et', 'also', 'state', 'It', 'values', 'T', 'p', 'problem', 'space', 'value', 'local', 'test', 'V', 'vector', 'k', 'f', 'b', 'may', 'performance', 'parameter', 'P', 'use', 'approach', 'new', 'B', 'examples', 'method', 'case', 'order', 'example', 'systems', 'shows', 'large', 'C', 'paper', 'based', 'eye', 'control', 'target', 'form', 'neurons', 'z', 'l', 'R', 'However', 'response', 'mean', 'current', 'memory', 'ensemble', 'neuron', 'patterns', 'M', 'u', 'small', 'r', 'task', 'c', 'To', 'layer', 'S', 'experts', 'many', 'well', 'class', 'cells', 'pattern', 'problems', 'process', 'methods', 'An', 'As', 'three', 'h', 'Systems', 'second', 'search', 'al', 'sets', 'result', 'found', 'H', 'expert', 'F', 'If', 'word', 'rate', 'show', 'work', 'term', 'average', 'D', 'Thus', 'terms', 'sequence']

filtered_words = [word for word in top_3 if word not in stopwords.words('english')]
print(filtered_words)
kickoff = [item.replace("()", "") for item in top_3]
# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = """To l 3 ca  "'' 01 00 (email)\ 7 de c g  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (713)[ ] I you am As it  can't  <<...>>  sincerely, .  > - < Kenneth Lay/Corp/Enron@Enron Best regards Sincerely From  Sent Original Message Q <-> * j  i'll |  \ /\ 100% 12345678910  (email)"' () """
# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords.split()
clean = [word for word in kickoff if word not in stoplist]
document=clean
tfidf_vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')

dtm = tfidf_vectorizer.fit_transform(document)  # sparse matrix with columns corresponding to words
tfidf_vectorizer.get_feature_names()

# Apply the vectoriser to the training set
Cardinality=0
for files in document:
    if files.endswith('.txt'):
        Cardinality+=1
counts = CountVectorizer(input='document')
dtm = counts.fit_transform(document)  # a sparse matrix
vocab = counts.get_feature_names()  # a list
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
matrix = tfidf_vectorizer.fit_transform(document)


vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

text = document
for token in text:
    syn_sets = wn.synsets(token)
    for syn_set in syn_sets:
        print(syn_set, syn_set.lemma_names())
        print(syn_set.hyponyms())
        print(syn_set.definition())
