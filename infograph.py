import re
import string
from nltk.tokenize import word_tokenize
freq={}



# load data
filename = 'jeff-dasovich_body.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words

tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
for word in words:
    count=freq.get(word,0)
    freq[word]=count + 1
frequency_list = freq.keys()
for words in frequency_list:
    print(words + ' -> ' + str(freq[words]))

results = []

for word in frequency_list:
    tuple = (word, freq[word])
    results.append(tuple)

byFreq=sorted(results, key=lambda word: word[1], reverse=True)
byFreq[:30]
words_names=[]
words_count=[]
for (word, freq) in byFreq[:20]:
    print (word, freq)
    words_names.append(word)
    words_count.append(freq)


import matplotlib.pyplot as plt
import numpy as np
# Plot histogram using matplotlib bar()
plt.xlabel('Top 30 Words')
plt.ylabel('Frequency')
plt.title('Plotting Word Frequency')
indexes = np.arange(len(words_names) )
width = .4
plt.bar(indexes, words_count, width)
plt.xticks(indexes + width * .1, words_names)
plt.legend()
plt.tight_layout()
plt.show()




