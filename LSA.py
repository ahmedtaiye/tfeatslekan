# LSA
import pandas as pd
import numpy as np
from scipy import linalg
import csv
#import set

edata = pd.read_csv("docword.enron.txt", skiprows=3, sep = ' ', header=None)
evocab = pd.read_csv("vocab.enron.txt", header=None)

evocab.columns  = ['word']
edata.columns = ['docid','wordid','freq']
# Taking a sample data set
edata = edata.iloc[:10000,:]

evocab.index = evocab.index + 1

wc = edata.groupby('wordid')['freq'].sum()



# Applying pivot


bag_of_words =edata.pivot(index='docid', columns='wordid', values='freq')

bag_of_words = bag_of_words.fillna(0)

sparse = bag_of_words.to_sparse(fill_value=0)

U,s,V = linalg.svd(sparse,full_matrices=False)

red_U = U[:,:200]
red_V = V[:200,:]
red_s = np.diag(s[:200])
reconstructedMatrix = np.dot(np.dot(red_U,red_s),red_V)

df_trans = pd.DataFrame(reconstructedMatrix,columns=bag_of_words.columns)

LSA = df_trans.apply(sum,0)





# from sklearn.decomposition import TruncatedSVD
#
# svd = TruncatedSVD(n_components=100,algorithm="randomized",n_iter=5,random_state=42)
#
# svd.fit(sparse)
#
# svd.explained_variance_ratio_.sum()
# svd.explained_variance_.shape
# X_new = svd.fit_transform(sparse)
#
# dim_red =  svd.transform(sparse)
#
# inv_trfm = svd.inverse_transform(dim_red)

def SET(m,in_list):
# Set the value of parameter m = the no. of iterations you require
    Card = pd.Series(np.NAN)
    DS=pd.Series(np.NAN)
    idx_added = pd.Series(np.NAN)
    pos = 0
    for j in range(1,m+1):
        np.random.seed(42+j)
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



# In[ ]:


# displaying the first 100 dissimilarities in decreasing order of smoothing factor.
#     cnt = 0
#     for i in df.sort_values(by=['Smoothing'], ascending=False).index:
#         if cnt < 100:
#             print(evocab.word[df.Index_added[i]] +"/n"+ df.DS[df.Index_added[i]])
#             cnt+=1
    A = df.sort_values(['Smoothing'],ascending=False)['Index_added'].head(1000)
    diss = df.sort_values(['Smoothing'],ascending=False)['DS'].head(1000)
    Smoothing = df.sort_values(['Smoothing'],ascending=False)['Smoothing'].head(1000)
    freq = wc[A]
    word = evocab.word[A]
    out = pd.DataFrame([word.values,freq.values,diss.values,Smoothing.values],index=['word','freq','diss','Smoothing']).transpose()
    #print(out)

    # find indexes of sets with max sf
    maxsf = []
    for i in range(len(df.DS)):
        if df.Smoothing[i] == df.Smoothing.max():
            maxsf.append(i)
    #print(maxsf)

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
    return out

out = SET(3,LSA)

print(out)

