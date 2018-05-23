from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
from nltk.corpus import stopwords

file = open("together.txt")
text = file.read()
# Apply the stoplist to the text
stoplist = stopwords.words('english')
clean = [word for word in text.split() if word not in stoplist]


#print (stoplist)

print (clean)

vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')


X = vectorizer.fit_transform(clean)
pd.DataFrame(X.toarray(),index=clean,columns=vectorizer.get_feature_names ()).head(10)

vectorizer.get_feature_names()

lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(X)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
lil= pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = vectorizer.get_feature_names())
lil2=pd.DataFrame(dtm_lsa, index = clean, columns = ["component_1","component_2" ])
print(lil, lil2)
xs = [w[0] for w in dtm_lsa]
ys = [w[1] for w in dtm_lsa]
xs, ys

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xs,ys)
xlabel=('First principal component')
ylabel=('Second principal component')
title=('Plot of points against LSA principal components')
plt.show()

import matplotlib.pyplot as plt
plt.figure()
ax = plt.gca()
ax.quiver(0,0,xs,ys,angles='xy',scale_units='xy',scale=1, linewidth = .01)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
xlabel=('First principal component')
ylabel=('Second principal component')
title=('Plot of points against LSA principal components')
plt.draw()
plt.show()

true_k = 1
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print ("Cluster %d:" % i,)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind],)
    print()
