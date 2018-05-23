
from __future__ import print_function
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
import pandas as pd
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
print ("\nCalculating document Dissimilarity and similarity scores...")
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
document = [ doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15]
# Set up the vectoriser, passing in the stop words

tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)
tfidf_matrix_train = tfidf_vectorizer.fit_transform(document)
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
vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
matrix = tfidf_vectorizer.fit_transform(document)
dtm = vectorizer.fit_transform(document)
t=vectorizer.get_feature_names()

#print (t)

x= ' '.join(t)
stop = set(stopwords.words('english'))
#sentence = "this is a foo bar sentence"
g=[i for i in x.lower().split() if i not in stop]
#print (nltk.pos_tag(g))
dd= ', '.join(str(x) for x in g)
#','.join(map(str,g) )
#print (dd)
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(dd)
#print (dd)
terms = vectorizer.get_feature_names()
from nltk.corpus import wordnet as wn
sents = dd
tokens = nltk.word_tokenize(dd)
tags = nltk.pos_tag(tokens)
nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'VBZ'or pos == 'VB')]
print(nouns)

lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
print (similarity)
#pd.DataFrame(similarity,index=nouns, columns=nouns).head(10)



true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
x=model.fit(matrix)
labels = x.labels_
print ("Top terms per labels:", labels)
print("Top terms per cluster:")
n_topics = 10

NUM_TOPICS = 10
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(dtm)
print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in different topic spaces
print(lsi_Z[0])

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Concepts %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


true_k = 10
print("LSI Model:")
print_topics(lsi_model, vectorizer)
print("=" * 20)

# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=10000, batch_size=10000, verbose=opts.verbose)

else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1, algorithm = 'arpack',
                verbose=opts.verbose)
#km.labels_=labels

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(matrix)
#km.labels_=labels
print ("Top terms per labels:",km.labels_)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(matrix, km.labels_, metric='cosine', sample_size=10000))
nbins=len(set(km.labels_))
vals,bins=np.histogram(km.labels_,bins=nbins)
print (20*' ','hist-min,max',np.min(vals),np.max(vals) )
print()


from sklearn.metrics import r2_score
from sklearn.metrics import v_measure_score
print(metrics.v_measure_score(labels, km.labels_))
print(r2_score(labels, km.labels_))

from sklearn.metrics import accuracy_score
print (accuracy_score(labels, km.labels_))














#word_list= nouns
#word_list = ['Jellicle', 'Cats', 'are', 'black', 'and', 'white,', 'Jellicle', 'Cats', 'are', 'rather', 'small;', 'Jellicle', 'Cats', 'are', 'merry', 'and', 'bright,', 'And', 'pleasant', 'to', 'hear', 'when', 'they', 'caterwaul.', 'Jellicle', 'Cats', 'have', 'cheerful', 'faces,', 'Jellicle', 'Cats', 'have', 'bright', 'black', 'eyes;', 'They', 'like', 'to', 'practise', 'their', 'airs', 'and', 'graces', 'And', 'wait', 'for', 'the', 'Jellicle', 'Moon', 'to', 'rise.', '']
word_counter = {}
for word in word_list:
     if word in word_counter:
         word_counter[word] += 1
     else:
         word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
top_ = popular_words[:100]


tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)
tfidf_matrix_train = tfidf_vectorizer.fit_transform(word_list)
# Apply the vectoriser to the training set
Cardinality=0
for files in document:
    if files.endswith('.txt'):
        Cardinality+=1
counts = CountVectorizer(input='word_list')
dtm = counts.fit_transform(tt)  # a sparse matrix
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
vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
matrix = tfidf_vectorizer.fit_transform(word_list)
#dtm = vectorizer.fit_transform(tt)
#vectorizer.get_feature_names()

#print (t)

#x= ' '.join(t)
stop = set(stopwords.words('english'))
#sentence = "this is a foo bar sentence"
#g=[i for i in x.lower().split() if i not in stop]
#print (nltk.pos_tag(g))
dd= ', '.join(str(x) for x in g)
''.join(word_list)
str1=''.join(str(e) for e in word_list)


true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
x=model.fit(matrix)
labels = x.labels_

lowercase = [x.lower() for x in word_list]
sents = lowercase
print (sents)


#wn.wup_similarity(sents, document)
from nltk.corpus import brown
freqs = nltk.FreqDist(w.lower() for w in sents)
print (freqs)


word_counter = {}
for word in dd:
     if word in word_counter:
         word_counter[word] += 1
     else:
         word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
top_ = popular_words[:100]

print (top_)






vectorizer = TfidfVectorizer(stop_words='english')
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=0.5, stop_words='english',
                                 use_idf=opts.use_idf)
dtm = vectorizer.fit_transform(df1)
print (dtm)
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
x=model.fit(dtm)
labels=x.labels_

l= vectorizer.get_feature_names()
print(l)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % dtm.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.

svd = TruncatedSVD(opts.n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(dtm)



import csv
with open('selected.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Colour'] == 'blue':
            print(row['ID'] ,row ['Make'],row ['Colour'])
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
Cardinality=0
for files in df1:
    if files.endswith('.txt'):
        Cardinality+=1
counts = CountVectorizer(input='df1')
dtm = counts.fit_transform(df1)  # a sparse matrix
vocab = counts.get_feature_names()  # a list
#type(dtm)
dtm = dtm.toarray()  # convert to a regular array

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
matrix = tfidf_vectorizer.fit_transform(df1)
#c = cosine_similarity(matrix)
