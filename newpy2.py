import pandas as pd
import numpy as np
import re
from nltk.stem import *
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

data = pd.read_csv("most common words in python.txt",sep=",",nrows=1)


reg_pat = '^[a-z][a-z\'-]+[a-z]$|^[a-z][a-z\'-]+[a-z]\.$|^[a-z][a-z\'-]+[a-z],$'
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

def pre_process(s):

    # convert all to lower case.
    s = s.lower()

    # Text is only being considered from "Subject" and "Body" of email.
    # to filter any lines that is due to forwards or replies in message.
    lines = filter(lambda line: not line.strip().startswith('>'),s.split('\n'))

    # to filter any lines that start from Date:,X-From,X-To,X-Folder,X-Origin,X-FileName.
    lines = filter(lambda line: not line.startswith('date:'),lines)
    lines = filter(lambda line: not line.startswith('x-from:'),lines)
    lines = filter(lambda line: not line.startswith('x-to:'),lines)
    lines = filter(lambda line: not line.startswith('x-folder:'),lines)
    lines = filter(lambda line: not line.startswith('x-origin:'),lines)
    lines = filter(lambda line: not line.startswith('x-filename:'),lines)

    # Tokenizing the Text message & considering only the tokens with length >= 3.
    arr = '\n'.join(lines).split()
    terms = []
    for term in arr:
        if re.match(reg_pat, term) != None:
            terms.append(term.replace(",","").replace(".",""))

    # Pruning the stop words.
    terms = list(filter(lambda term: term not in stop, terms))

    # Perform Stemming on terms.
    # terms = list(pd.Series(terms).map(stemmer.stem))

    return terms

lsa_data = data
#print(lsa_data)									# display enron data
temp = []
for i in range(len(data)):
	count = 0
	for j in range(len(lsa_data)-1):
		if lsa_data[0][i] in lsa_data[j+1]:		# delete repeatable data from lsa_data list
			count += 1
	if count == len(lsa_data)-1:
		temp.append(lsa_data[0][i])				# if end of list, insert into temp list

for i in range(len(temp)):
	for ttt in lsa_data:
		ttt.remove(temp[i])

print ("========reral")
print (lsa_data)

word_dict={}
for i in lsa_data:
    for word in i:
        word_dict[word] = word_dict.get(word,0)+1
word_dict

words=[]
countli=[]
for key in word_dict:
    words.append(key)
    countli.append(word_dict[key])

def SET(m,in_list):
# Set the value of parameter m = the no. of iterations you require
    Card = pd.Series(np.NAN)
    DS=pd.Series(np.NAN)
    idx_added = pd.Series(np.NAN)
    pos = 0
    for j in range(1,m+1):
        new_indices = np.random.choice(in_list.index,len(in_list),replace=False)
        for i in pd.Series(new_indices).index:
            idx_added[i+pos] = new_indices[i]
            DS[i+pos]=np.var(in_list[new_indices[:i+1]])
            Card[i+pos] = len(in_list[:i+1])
        pos = pos+i+1

    df = pd.DataFrame({'Index_added':idx_added,'DS':DS,'Card':Card})
    df ['DS_Prev'] = df.DS.shift(1)
    df['Card_prev'] = df.Card.shift(1)
    df.Card_prev[(df.Card == 1)] = 0
    df = df.fillna(0)
    df['Smoothing'] = (df.Card - df.Card_prev)*(df.DS - df.DS_Prev)


    # find indexes of sets with max sf
    maxsf = []
    for i in range(len(df.DS)):
        if df.Smoothing[i] == df.Smoothing.max():
            maxsf.append(i)
    print(maxsf)

    N = len(in_list)
    excp_set = []
    for i in range(len(maxsf)):
        j = maxsf[i]
        k=j+1
        temp = []
        temp.append(df.Index_added[j])
        excp_set.append(temp.copy())
        temp_prev = pd.Series()
        temp_j = pd.Series()
        a=j
        while(a%N!=0):
            temp_j.set_value(len(temp_j),in_list[df.Index_added[a]])
            a=a-1
        temp_j.set_value(len(temp_j),in_list[df.Index_added[a]])   # Ij
        temp_prev = temp_j.copy()                   # Ij-1
        del(temp_prev[0])
        temp_prev.index = np.arange(len(temp_prev))
        while(k%N!=0):
            K_element = in_list[df.Index_added[k]]    # K th element
            temp_prev.set_value(len(temp_prev),K_element)            # Ij-1 U {ik}
            temp_j.set_value(len(temp_j),K_element)               # Ij U {ik}
            Dk0 = np.var(temp_prev) - df.DS[j-1]
            Dk1 = np.var(temp_j) - df.DS[j]
            if Dk0-Dk1 >= df.DS[j]:                # If Dk0 - Dk1 >= Dj we add the element to exception set.
                excp_set[i].append(df.Index_added[k])
            del(temp_prev[len(temp_prev)-1])
            del(temp_j[len(temp_j)-1])
            k+=1
    #print(excp_set)                                # contains the indices of exception elements.
    return excp_set
excp_set = SET(2,pd.Series(countli))

# Printing the exception words.
print("\nException set: \n")
for i in range(len(excp_set)):
    print(pd.Series(words)[excp_set[i]])

