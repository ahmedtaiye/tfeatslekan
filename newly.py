import logging
from pprint import pprint
#logging.basicConfig( format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

from gensim import corpora, models, similarities
documents = ["enron email business Ken lay information meeting communication",
             "enron Jeff Skillings email subject enron.com message",
             "enron, Energy power market california email gas",
             "Kay mann enron subject enron.co agreement message form",
             "power enron california energy million email electricity ",
             "enron Tama Jone communication credit trading forwarded counterpart",
             "enron power energy california market electricity company",
             "enron Sara Shackleton subject enron.com agreement forwarded mail",
             "enron company subject message dynergy original market",
             "enron dana email enron.com message forwarded",
             "enron Mark Taylor subject legal trading email forwarded",
             "Chris subject gas deal enron.com capacity storage",
             "Power energy Kate symmes deal enron subject California"]

pprint( len( documents ))

stoplist = set( 'for of a the and to in'.split() )
texts = [ [word for word in document.lower().split() if word not in stoplist ] for document in documents ]
pprint( texts )

from collections import defaultdict
frequency = defaultdict( int )
for text in texts:
    for token in text:
        frequency[ token ] += 1
pprint( frequency )

texts2 = [ [ token for token in text if frequency[ token ] > 1 ] for text in texts ]

pprint( texts2 )
dictionary = corpora.Dictionary( texts2 )
print( dictionary )
print( dictionary.token2id )

newdoc = 'agreement form legal meeting'
newvec = dictionary.doc2bow( newdoc.split() )


corpus = [ dictionary.doc2bow( text ) for text in texts ]
corpora.MmCorpus.serialize( './deerwster.mm', corpus )
corpora.SvmLightCorpus.serialize('./corpus.svmlight', corpus)
tfidf = models.TfidfModel( corpus )
corpus_tfidf = tfidf[ corpus ]
for doc in corpus_tfidf:
    print( doc )
# latent semantic analysis
lsi = models.LsiModel( corpus, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity( lsi[ corpus ] )
veclsi = lsi[ newvec ]
print( '+'*20 )
print( veclsi )

sims = index[ veclsi ]
for i, sim in enumerate( sims):
    pprint( documents[i]+":::"+newdoc+" similarity_score_is {}".format( sim )  )

print( "+"*20 )
print( "+"*20 )
sims2 = index[  lsi[ dictionary.doc2bow(texts2[0])] ]
for i, sim in enumerate( sims2):
    pprint( documents[i]+":::"+documents[0]+" similarity_score_is {}".format( sim )  )
