import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
import nltk
#nltk.download('punkt')
import re
import time
import codecs
import seaborn as sns
import matplotlib.gridspec as gridspec

from subprocess import check_output
#print(check_output(["NIPS"]).decode("utf8"))

def clean_text(text):
    list_of_cleaning_signs = ['\x0c', '\n']
    for sign in list_of_cleaning_signs:
        text = text.replace(sign, ' ')
    #text = unicode(text, errors='ignore')
    clean_text = re.sub('[^a-zA-Z]+', ' ', text)
    return clean_text.lower()
from nltk.corpus import stopwords
# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = """To  [ ] I you am As it  can't  <<...>>  sincerely, .  > - < Kenneth Lay/Corp/Enron@Enron Best regards Sincerely From  Sent Original Message Q <-> * | /\ 100% 12345678910 () """

# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords.split()

stopWords = stopwords.words('english')
authors = pd.read_csv("authors.csv")
paper_authors = pd.read_csv("paper_authors.csv")
papers = pd.read_csv("pap.csv")


g = sns.countplot(papers.year)
plt.xticks(rotation=90)
plt.show()
