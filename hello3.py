from __future__ import print_function

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
print (doc1)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
with open("20newsgroupdatasetallterms2.txt", "r") as f:

    data = f.read()
words= word_tokenize(data)
useful_words = [word  for word in words if word not in stopwords.words('English')]
frequency = nltk.FreqDist(useful_words)
print(frequency.most_common(100))
