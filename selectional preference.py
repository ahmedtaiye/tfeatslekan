from nltk.tag import pos_tag
import re
import nltk
from nltk.corpus import wordnet as wn
import math
import numpy as np
from scipy import spatial
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import nltk
stopwords = nltk.corpus.stopwords.words(fileids='english')
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.metrics import edit_distance
file = open("most common words in python.txt")
sentence = file.read()
tagged_sent = pos_tag(sentence.split())
print (tagged_sent )
# [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]

propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
# ['Michael','Jackson', 'McDonalds']
#print (propernouns)
possessives = [word for word in sentence if word.endswith("'s") or word.endswith("s'")]

Nouns = [word for word,pos in tagged_sent if pos == 'NNP']
Verbs = [word for word,pos in tagged_sent if pos == 'VBZ']
verb = [word for word,pos in tagged_sent if pos == 'VB']
x=(possessives,propernouns,Nouns)
print(possessives,propernouns,Nouns)

#jj=('Enron,', 'To,', 'California,', 'Ken,', 'Lay,', 'Please,', 'Millions,', 'Funds,', 'Retirement', 'Houston,', 'New', 'York,', 'Money,', 'May,', 'Plans,', 'Time,')
from nltk.corpus import wordnet
from nltk.corpus import wordnet
import itertools as IT
list1 = ["apple", "honey"]
list2 = ["apple", "shell", "movie", "fire", "tree", "candle"]

for word1, word2 in IT.product(list1, list2):
    wordFromList1 = wordnet.synsets(word1)[0]
    wordFromList2 = wordnet.synsets(word2)[0]
    s = wordFromList1.wup_similarity(wordFromList2)
    print('{w1}, {w2}: {s}'.format(w1 = wordFromList1.name,w2 = wordFromList2.name,s = wordFromList1.wup_similarity(wordFromList2)))



