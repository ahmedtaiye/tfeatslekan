
from __future__ import print_function
from sklearn.metrics.pairwise import cosine_similarity
import nltk
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
matrix = tfidf_vectorizer.fit_transform(document)


vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(document)
c = cosine_similarity(document)
print ("\nSimilarity Score [*] ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train))
#print (c)

stri= ""    #create empty string to manipulate data
for line in document:
    stri+=line
word_stri = stri.split()    #split the string and convert it into list
word_counter = {}
for word in word_stri:
     if word in word_counter:
         word_counter[word] += 1
     else:
         word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
top_3 = popular_words[:100]
print (top_3)

x= ' '.join(top_3)
stop = set(stopwords.words('english'))
#sentence = "this is a foo bar sentence"
g=[i for i in x.lower().split() if i not in stop]
#print (nltk.pos_tag(g))
dd= ', '.join(str(x) for x in g)
#','.join(map(str,g) )
#print (dd)
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(dd)
print (dd)
terms = vectorizer.get_feature_names()
from nltk.corpus import wordnet as wn
sents = dd


tokens = nltk.word_tokenize(dd)
tags = nltk.pos_tag(tokens)
nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'VBZ'or pos == 'VB')]
print(nouns)

nouns2= ', '.join(str(nouns) for x in g)


data_vectorized2 = vectorizer.fit_transform(nouns2)

pd.DataFrame(dtm.toarray(),index=nouns2,columns=vectorizer.get_feature_names ()).head(10)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
x=model.fit(matrix)
labels = x.labels_
print("Top terms per cluster:")
n_topics = 10

NUM_TOPICS = 10
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(data_vectorized2)
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
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1, algorithm = 'arpack',
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(matrix)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(matrix, km.labels_, sample_size=10000))

print()





dd2= ', '.join(str(dd) for x in g)
stri= ""    #create empty string to manipulate data
for line in dd2:
    stri+=line
word_stri = stri.split()    #split the string and convert it into list
word_counter = {}
for word in word_stri:
     if word in word_counter:
         word_counter[word] += 1
     else:
         word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
top_3 = popular_words[:100]
print (top_3)



    listSimilarities = sum(similarities)
    listLength= len(similarities)
    listAverage= (listSimilarities / listLength)
    np.mean(listAverage)






    #print(x)
    #np.mean(similarities)

    #print (np.mean(tt))


    #np.mean(similarities2)
    #print (synsets)

from nltk.metrics import *
#print(accuracy( similarities, similarities2))
#print (precision( similarities, similarities2))
#print(recall(similarities, similarities2))
#print()


from sklearn.neighbors import BallTree as BallTree
BT = BallTree(similarities, leaf_size=5, p=2)
dx, idx = BT.query(similarities[500,:], k=3)
print (BT.query(dx, idx = BT.query(similarities[500,:], k=3)[500,:], k=3))



tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)
# Apply the vectoriser to the training set
Cardinality=0
for files in document:
    if files.endswith('.txt'):
        Cardinality+=1
counts = CountVectorizer(input='nouns')
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
#c = cosine_similarity(matrix)
#print ("\nSimilarity Score [*] ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train))
#print (c)
true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
x=model.fit(matrix)
labels = x.labels_

lowercase = [x.lower() for x in nouns]
sents = lowercase

synsets = [synset
           for word in terms
           for synset in wn.synsets(word, 'n')]
for s in synsets:
    similarities = [s.path_similarity(t)*1 for t in synsets]
    similarities2= [s.lch_similarity(t)*1 for t in synsets]
    similarities3= [s.wup_similarity(t)*1 for t in synsets]
    similarities4= [s.wup_similarity(t)*1 for t in synsets]
    row  = ' '.join('{:3.0f}'.format(s) for s in similarities)
    row2 = ' '.join('{:3.0f}'.format(s) for s in similarities2)
    row3 = ' '.join('{:3.0f}'.format(s) for s in similarities3)
    x='{:2} {}'.format(s.name(), row)






lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

true_k = 5
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
km.fit(dtm)
#labels = x.labels_
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names()
for i in range(true_k):
     print("Cluster %d:" % i, end='')
     for ind in order_centroids[i, :10]:
         print(' %s' % t[ind], end='')
         print()


print (t[ind])


synsets = [synset
           for word in word_list
           for synset in wn.synsets(word, 'n')]
for s in synsets:
    similarities = [s.path_similarity(sents)*10 for word_list in synsets]
    similarities2= [s.lch_similarity(t)*10 for t in synsets]
    similarities3= [s.wup_similarity(t)*10 for t in synsets]
    similarities4= [s.wup_similarity(t)*10 for t in synsets]
    row  = ' '.join('{:3.0f}'.format(s) for s in similarities)
    row2 = ' '.join('{:3.0f}'.format(s) for s in similarities2)
    row3 = ' '.join('{:3.0f}'.format(s) for s in similarities3)
    x='{:2} {}'.format(s.name(), row)


sents = top_3
tokens = nltk.word_tokenize(sents)
tags = nltk.pos_tag(tokens)

nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS'or pos == 'VB' or pos == 'VBZ' or pos == 'd' or pos == 's')]
word_list= nouns
df1 = pd.read_csv("selected.csv", header=None)
print(df1)


data=fetch_20newsgroups()
dataset = fetch_20newsgroups(subset='all', random_state=42)
import pandas as pd
df = pd.read_csv("selected.csv", sep=",")
output = pd.DataFrame(columns=df.columns)
for c in df.columns:
    if df[c].dtype == object:
        print ("convert ", df[c].name, " to string")
        c = df.select_dtypes(include=[object]).columns
        df[c] = df[c].astype(str)

    output.to_csv("selected.csv_tostring2.csv", index=False)



#print (data)

print("Extracting features from the dataset using a sparse vectorizer")
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
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
x=vectorizer.fit_transform(data)
y = data.target




true_k = np.unique(y).shape[0]
svd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
normalizer = Normalizer(copy=True)
lsa = make_pipeline(svd, normalizer)
print (lsa)
dtm= lsa.fit(x)
print (x)
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

print()
# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=100000, batch_size=100000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(x)
print("done in %0.3fs" % (time() - t0))
print()
print(svd.explained_variance_ratio_)


print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(y, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(y, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(y, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x, km.labels_, sample_size=1000000))

print()
