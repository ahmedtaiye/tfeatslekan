from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
import logging
import pandas as pd
import warnings
import numpy as np
f = open('nawao2.txt')
doc1 = str(f.read())
vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
input=[input]
dtm = vectorizer.fit_transform(doc1)
pd.DataFrame(dtm.toarray(),index=doc1,columns=vectorizer.get_feature_names ()).head(10)
lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = vectorizer.get_feature_names())
print(pd.DataFrame(dtm_lsa, index = doc1, columns = ["component_1","component_2" ]))

xs = [w[0] for w in dtm_lsa]
ys = [w[1] for w in dtm_lsa]
xs, ys

#Plot scatter plot of points %pylab inline
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xs,ys)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Plot of points against LSA principal components')


for word in string.replace(',', ' ').split():
  word

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

word_list = word

counts = dict(Counter(word_list).most_common(10))

labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

bar_width = 0.35

plt.bar(indexes, values)

# add labels
plt.xticks(indexes + bar_width, labels)
plt.show()



import numpy as np
import sklearn.cluster
import distance

words = strr #Replace this line
words = np.asarray(words) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))
