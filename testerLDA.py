from __future__ import print_function
import sklearn

sklearn.__version__

import os

import nltk
import sklearn
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import sklearn.feature_extraction.text as text
from sklearn import decomposition

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

filenames = ['ken-lay_body.txt',
             'jeff-skilling_body.txt',
             'Richard-shapiro_body.txt',
             'kay-mann_body.txt',
             'Jeff-dasovich_body.txt',
             'tana jones_body.txt',
             'steven kean_body.txt',
             'shackleton sara_body.txt',
             'james steffes_body.txt',
             'Mark taylor_body.txt',
             'davis pete_body.txt',
             'Chris g_body.txt',
             'kate symes_body.txt']

#Calculating the cardinality and dissimilarities.
Cardinality=0
for files in filenames:
    if files.endswith('.txt'):
        Cardinality+=1

vectorizer = CountVectorizer(input='filename')


dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
vocab = vectorizer.get_feature_names()  # a list
type(dtm)

dtm = dtm.toarray()  # convert to a regular array
vocab = np.array(vocab)
n, _ = dtm.shape

dist = np.zeros((n, n))
Dissimilarity=dist
for i in range(n):
     for j in range(n):
        x, y = dtm[i, :], dtm[j, :]
        dist[i, j] = np.sqrt(np.sum((x - y)**2))

len(vocab)

num_topics = 13
num_top_words = 13
clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)

# print words associated with topics
topic_words = []
for topic in clf.components_:
     word_idx = np.argsort(topic)[::-1][0:num_top_words]
     topic_words.append([vocab[i] for i in word_idx])

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
novel_names = []
for fn in filenames:
     basename = os.path.basename(fn)
     name, ext = os.path.splitext(basename)
     name = name.rstrip('0123456789')
     novel_names.append(name)


novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()
num_groups = len(set(novel_names))
doctopic_grouped = np.zeros((num_groups, num_topics))

for i, name in enumerate(sorted(set(novel_names))):
     doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
doctopic = doctopic_grouped
print(doctopic)

novels = sorted(set(novel_names))
print("Top NMF topics in...")
for i in range(len(doctopic)):
     top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
     top_topics_str = ' '.join(str(t) for t in top_topics)
     print("{}: {}".format(novels[i], top_topics_str))
# show the top 15 words
for t in range(len(topic_words)):
     print("Topic {}: {}".format(t, ' '.join(topic_words[t][:30])))




