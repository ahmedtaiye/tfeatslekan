
from __future__ import print_function
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
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
# Apply the vectoriser to the training set
Cardinality=0
for files in document:
    if files.endswith('.txt'):
        Cardinality+=1
counts = CountVectorizer(input='train_set')
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
c = cosine_similarity(matrix)
#print ("\nSimilarity Score [*] ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train))
#print (c)
true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
x=model.fit(matrix)
labels = x.labels_
print("Top terms per cluster:")
n_topics = 10

NUM_TOPICS = 10
vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(document)

# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(data_vectorized)
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

terms = vectorizer.get_feature_names()



tagged_sent = pos_tag(document)
#print (tagged_sent )
# [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]

propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
# ['Michael','Jackson', 'McDonalds']
#print (propernouns)
possessives = [word for word in document if word.endswith("'s") or word.endswith("s'")]
Nouns = [word for word,pos in tagged_sent if pos == 'NNP']
Verbs = [word for word,pos in tagged_sent if pos == 'VBZ']
verb = [word for word,pos in tagged_sent if pos == 'VB']
det= [word for word,pos in tagged_sent if pos == 'd']
y=(possessives,propernouns,Nouns,Verbs, verb, det)
#print(possessives)
#print(y)
sents = ['Enron,', 'company,', 'employees', 'energy,', 'To,', 'made,', 'California,', 'Ken,', 'Lay,', 'Please,', 'Millions,', 'stock', 'Funds,', 'Retirement', 'bills,', 'bankruptcy,', 'donate,', 'donate,', 'declared,', 'year,', 'help,', 'information,', 'provide,', 'this,', 'subject,', 'like,', 'business,', 'meeting,', 'Houston,', 'New', 'York,', 'enron.com,', 'Money,', 'october,', 'result,', 'largest,', 'May,', 'Plans,', 'Time,', 'communications.']
synsets = [synset
           for word in terms
           for synset in wn.synsets(word, 'n')]
for s in synsets:
    similarities = [s.path_similarity(t)*10 for t in synsets]
    row = ' '.join('{:3.0f}'.format(s) for s in similarities)
    x='{:2} {}'.format(s.name(), row)
    #print(x)
    for synset in wn.synsets (sents):
        print(synset, sents.definition())
print(y)




# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=100, batch_size=100, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
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
      % metrics.silhouette_score(matrix, km.labels_, sample_size=10))

print()


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

sents = dd



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
__doc__
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
print ("\nESET accessing datasets with scores and dissimilarities...")
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
# Set up the vectori
#print (document)

ee= "Enron, Subject, PM, would, enron.com, ENRON, We, Please, Thanks, time, said, Energy, Forwarded, This, need, get, one, E-mail, may, information, power,new, like, call, Jeff, Mark, market, could, deal, California, let, business, New, John, Houston, company, think, see, meeting, make, energy, Sent, ENRON_DEVELOPMENT, Mike, Communications, day, Chris, questions, last"
tt=ee.split(' ')




x= ' '.join(document)
stop = set(stopwords.words('english'))
g=[i for i in x.lower().split() if i not in stop]
#print (nltk.pos_tag(g))
dd= ', '.join(str(x) for x in g)
#print (dd)



stri= ""    #create empty string to manipulate data
for line in dd:
    stri+=line
word_stri = stri.split()    #split the string and convert it into list
word_counter = {}
for word in word_stri:
     if word in word_counter:
         word_counter[word] += 1
     else:
         word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
#top_3 = popular_words[:1000]
