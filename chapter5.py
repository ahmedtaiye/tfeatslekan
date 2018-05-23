from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#import codecs, difflib, Levenshtein, distance

import numpy as np

# Bring in standard stopwords
stopWords = stopwords.words('english')

print ("\nCalculating document similarity scores...")

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
train_set = [ doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15]

# Set up the vectoriser, passing in the stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)

# Apply the vectoriser to the training set
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

# Print the score
print ("\nSimilarity Score [*] ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train))
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_matrix_train1 = tfidf_vectorizer.fit_transform(train_set)
D = euclidean_distances(tfidf_matrix_train1)
print ("\nEuclideanSimilarity Score [*] ",euclidean_distances(tfidf_matrix_train[0:1], tfidf_matrix_train))

#print ("\nSimilarity Score [*] ",jaccard_similarity_score(tfidf_matrix_train[0:1], tfidf_matrix_train)
from sklearn.metrics.pairwise import manhattan_distances
E= manhattan_distances(tfidf_matrix_train1)
print ("\nManhattan_distancesSimilarity Score [*] ",1-(manhattan_distances(tfidf_matrix_train[0:1], tfidf_matrix_train))/100)
from sklearn.metrics.pairwise import cosine_distances
E= 1-cosine_distances(tfidf_matrix_train1)
print ("\nCosine_distancesSimilarity Score [*] ",cosine_distances(tfidf_matrix_train[0:1], tfidf_matrix_train))

from sklearn.metrics.pairwise import pairwise_distances
jac_sim = 1 - pairwise_distances(tfidf_matrix_train1)
print ("\nJac_SimSimilarity Score [*] ",cosine_distances(tfidf_matrix_train[0:1], tfidf_matrix_train))
