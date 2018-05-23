from __future__ import print_function
import sklearn
import mpl_toolkits
import os  # for os.path.basename
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
import logging
import os, sys

from optparse import OptionParser
import sys
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
f = open('nawao2.txt', encoding="ISO-8859-1")
filenames = str(f.read())
#print(filenames)
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(filenames)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

word_tokens
filtered_sentence
str1 = ''.join(filtered_sentence)
#print(str1)
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
x=tokenizer.tokenize(str1)
#print (x)


blacklist = ['(',')','Î±k','Î²k', '=', '2', 'Î±kâ\x88\x921', '+',  '[', '7', ']', ',',  'MH','â\x88¥xi', 'â\x88¥2','&', ',', 'Y', '=', 'â\x88¥yi', 'â\x88¥2','n1', ',', 'n2', 'X', '='
             'â\x87', 'â\x9c\x93', 'â\x87', 'i', 'j', 'â\x87', 'N', 'â\x86µ', 'Î³', '=', '0.001', ',', 'of', 'Ï\x810', '=', '1e', 'â\x88\x92','0.8','0.05','MCMC','Carrollâ\x80\x99s',
             'Aliceâ\x80\x99s','.','6', 'â\x88\x88', '{', '0', '1', '}', 'i.e', '.','Î', '»',  'â\x88\x80E', 'GE', 'â\x88\x80u','â\x88\x80E', 'â\x86\x92', 'â\x86\x92', 'W', ';', '5', 'â\x88\x80V'  ]
cleaned = []
for item in word_tokens:
    clean = True
    for exclude in blacklist:
        if item.find(exclude) != -1:
            clean = False
            break
    if clean:
        cleaned.append(item)
print (cleaned)

import re
x=list(filter(lambda x:x, map(lambda x:re.sub(r'[^A-Za-z]', '', x), cleaned)))
string = " ".join(x)

print (string)


from collections import Counter
words= cleaned
most_common_words= [word for word, word_count in Counter(words).most_common(200)]
print (most_common_words)

import re
xx=list(filter(lambda x:x, map(lambda x:re.sub(r'[^A-Za-z]', '', x), most_common_words)))
from stop_words import get_stop_words
from nltk.corpus import stopwords

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)

output = [w for w in xx if not w in stop_words]
print(output)
string = " ".join(xx)
#print (string)

stop_words = set(["the", "and", "that", 'for',"by", "to","from", "as", "on", "be","are","we","x","t", "can", "The", "an", "at","n"])

strr=" ".join(word for word in string.split() if word not in stop_words)


doc1=strr.split()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(doc1)

true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.gridspec as gridspec
from subprocess import check_output
#print(check_output('paper_authors.csv').decode("utf8"))

authors = pd.read_csv("authors.csv")
paper_authors = pd.read_csv("paper_authors.csv")
#papers = pd.read_csv("pap.csv")
g = sns.countplot(authors.year)
plt.xticks(rotation=90)

authors_new = authors.rename(columns = {'id':'author_id'})
paper_authors_new = pd.merge(paper_authors, authors_new, on='author_id', how='left')
