from __future__ import print_function
import sklearn
import pandas as pd
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
sklearn.__version__
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
from nltk.corpus import stopwords

# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = """To  [ ] I you am As it  can't  <<...>>  sincerely, .  > - < Kenneth Lay/Corp/Enron@Enron Best regards Sincerely From  Sent Original Message Q <-> * | /\ 100% 12345678910 () """

# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords.split()

# Open a file and read it into memory

# Open a file and read it into memory
file = open('most common words in python.txt')
text = file.read()


# Apply the stoplist to the text
clean = [word for word in text.split() if word not in stoplist]
stoplist = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
#print (stoplist)
stoplist += additional_stopwords.split()
#print (clean)
#print (', '.join(clean))
#print ('\n'.join(clean))
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in clean:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
#print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(clean)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


#print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    #print ("Cluster %d:" % i,)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind],)

    #print()
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
file = open("most common words in python.txt")
sentence = file.read()
tagged_sent = pos_tag(sentence.split())
#print (tagged_sent )
# [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]

propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
# ['Michael','Jackson', 'McDonalds']
#print (propernouns)
possessives = [word for word in sentence if word.endswith("'s") or word.endswith("s'")]
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

# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Build a dataframe with 4 connections
df = pd.DataFrame({ 'from':['POWER', 'ENERGY', 'POWER', 'ENRON','POWER','MEETING','POWER','COMPANY','POWER','ELECTRICITY','ENRON', 'ENERGY', 'ENERGY', 'COMPANY', 'ENRON', 'BUSINESS','ENRON', 'BANK', 'ENERGY', 'BANK', 'BUSINESS', 'ENERGY', 'BUSINESS', 'MONEY', 'COMPANY', 'MONEY', 'MONEY', 'FUND', 'FUND', 'BUSINESS', 'PLAN', 'COMPANY','PLAN', 'BUSINESS', 'EMPLOYEE','ENRON', 'EMPLOYEE','BUSINESS', 'EMPLOYEE','COMPANY', 'EMPLOYEE','MONEY', 'RETIREMENT','PLAN', 'KENLAY','ENRON', 'RETIREMENT', 'EMPLOYEE', 'RETIREMENT','MONEY','ENRON','YEAR', 'TIME', 'ENRON','OCTOBER', 'TIME', 'OCTOBER', 'YEAR', 'ENRON','CALIFORNIA','ENRON','HOUSTON', 'COMPANY', 'STOCK','STOCK', 'MONEY', 'NEW YORK', 'ENRON', 'MILLION', 'MONEY', 'COMMUNICATION', 'BUSINESS', 'COMMUNICATION', 'INFORMATION', 'COMMUNICATION','ENRON', 'INFORMATION', 'SUBJECT', 'INFORMATION','ENRON','BILL', 'INFORMATION','COMMUNICATION','BILL', 'ENRON','BILL','BANK','BILL', 'MEETING','KENLAY', 'AID','BANK', 'MILLION', 'STOCK', 'MEETING','INFORMATION', 'COMMUNICATION','MEETING','INFORMATION','CALIFORNIA','INFORMATION','NEW YORK','INFORMATION', 'HOUSTON', 'ENRON','AID','BILL','BUSINESS','SUBJECT','COMMUNICATION','SUBJECT','ENRON','EMPLOYEE','INFORMATION','BANK','INFORMATION','COMPANY','INFORMATION','ENERGY','INFORMATION','BANK','EMPLOYEE','BANK','BILL','TIME','YEAR','TIME','MEETING','TIME','FUND','TIME','AID','TIME','FUND','TIME','STOCK','TIME','PLAN','TIME','COMMUNICATION','TIME','INFORMATION','TIME','RETIREMENT','RETIREMENT','YEAR'],
                    'to':['COMPANY','POWER','COMPANY','ELECTRICITY','ENERGY','POWER','ENRON','POWER','ENERGY','MEETING','POWER', 'ENRON', 'COMPANY', 'ENERGY','BUSINESS', 'ENRON', 'BANK', 'ENRON', 'BANK', 'ENERGY', 'ENERGY', 'BUSINESS', 'MONEY', 'BUSINESS', 'MONEY', 'COMPANY', 'FUND', 'MONEY', 'BUSINESS', 'FUND', 'COMPANY','PLAN', 'BUSINESS', 'PLAN','ENRON', 'EMPLOYEE','BUSINESS', 'EMPLOYEE', 'COMPANY', 'EMPLOYEE','MONEY','EMPLOYEE', 'PLAN', 'RETIREMENT', 'ENRON', 'KENLAY','EMPLOYEE','RETIREMENT', 'MONEY', 'RETIREMENT','YEAR', 'ENRON', 'ENRON', 'TIME', 'TIME', 'OCTOBER', 'YEAR', 'OCTOBER', 'CALIFORNIA', 'ENRON', 'HOUSTON','ENRON', 'STOCK','COMPANY','MONEY','STOCK','ENRON','NEW YORK', 'MONEY', 'MILLION', 'BUSINESS', 'COMMUNICATION','INFORMATION', 'COMMUNICATION', 'ENRON', 'COMMUNICATION','SUBJECT', 'INFORMATION', 'ENRON', 'INFORMATION','INFORMATION', 'BILL', 'BILL','COMMUNICATION','BILL','ENRON','BILL','BANK', 'KENLAY', 'MEETING','BANK','AID','STOCK', 'MILLION','INFORMATION', 'MEETING','MEETING','COMMUNICATION','CALIFORNIA', 'INFORMATION','NEW YORK', 'INFORMATION','HOUSTON', 'INFORMATION','AID','ENRON','BUSINESS','BILL','COMMUNICATION','SUBJECT','ENRON','SUBJECT','INFORMATION','EMPLOYEE','INFORMATION','BANK','INFORMATION','COMPANY','INFORMATION','ENERGY','EMPLOYEE','BANK','FUND','BANK','FUND','TIME','BILL','TIME','YEAR','TIME','MEETING','TIME', 'AID','TIME', 'STOCK','TIME','PLAN','TIME','COMMUNICATION','TIME','INFORMATION','TIME','RETIREMENT','TIME','YEAR','RETIREMENT']})
df

# Build your graph
G=nx.from_pandas_dataframe(df, 'from', 'to')

# Plot it
names = [os.path.basename(fn).replace('.txt', '') for fn in df]
color = 'red' if "ENRON" in names else 'skyblue'
nx.draw(G, with_labels=True, c=color)

plt.show()


