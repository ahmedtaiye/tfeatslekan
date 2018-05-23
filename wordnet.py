from __future__ import print_function
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import logging
import numpy as np
from itertools import product
from nltk.corpus import wordnet as wn
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
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
data = [ doc1, doc2, doc3, doc4, doc5, doc6,doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15]
top_3= ['subject', 'power', 'would', 'cc', 'enron', 'pm', 'please', 'energy', 'new', 'said', 'may', 'california', 'forwarded', 'state', 'also', 'gas', 'one', 'know', 'need', 'get', 'could', '&', '', 'market', 'price', 'company', 'electricity', 'call', 'like', 'us', 'information', '', 'business', 'attached', 'let', 'agreement', 'time', 'mark', 'last', 'jeff', '', '', 'million', '?', 'two', 'make', 'trading', 'first', 'see', 'sara', 'credit', 'use', 'meeting', 'corp', 'deal', 'think', 'prices', 'next', 'thanks', 'financial', 'group', '', 'kay', 'want', 'john', 'vince', 'take', 'per', 'email', 'legal', 'companies', 'back', 'rate', 'contact', 'utilities', 'forward', 'president', 'bill', 'services', 'following', 'customers', 'going', 'said', 'san', 'message', 'work', 'service', 'sent', 'provide', 'send', 'public', 'day', 'north', 'davis', 'access', 'much', 'go', 'still', 'since', 'order', 'received', 'david', 'inc', 'tana', 'thanks', 'utility', 'ferc', 'commission', 'electric', 'help', 'contract', "i'm", 'plan', 'report', '0909', 'number', 'management', 'made', 'email', 'good', 'office', 'copy', 'trade', 'review', 'issues', 'federal', 'natural', 'conference', 'many', 'look', 'america', 'billion', 'find', 'way', 'comments', 'set', 'draft', 'thank', 'firm', 'change', 'jones', 'week', 'april', 'available', 'system', 'give', 'people', 'risk', 'mr', 'part', 'year', 'sent', 'contracts', 'changes', 'online', 'issue', 'working', 'must', 'mike', 'message', 'well', 'current', 'original', 'list', 'letter', 'regarding', 'pay', 'shall', 'project', 'paul', 'isda', 'board', 'global', 'houston', 'however', 'questions', 'support', 'even', 'including', 'texas', 'whether', 'cost', 'product', 'transmission', 'chris', 'pg&e', 'news', 'ena', 'wholesale', 'form', 'three', 'name', 'internet', 'based', "state's", 'best', 'june', 'end', 'plant', 'might', 'put', 'richard', 'already', 'high', 'free', 'government', 'sure', 'thanks', 'additional', 'march', 'percent', 'committee', 'development', 'edison', 'master', '(email)\'"', 'include', 'today', 'supply', 'capital', 'discuss', 'date', 'request', 'buy', 'susan', 'asked', 'able', 'proposed', 'rates', 'regulatory', 'plants', '2', '', 'final', 'another', 'us', 'customer', 'dow', 'james', "california's", 'address', '1', 'executive', 'general', 'right', 'used', 'several', 'click', 'costs', 'open', 'demand', 'technology', 'come', 'michael', 'hope', 'keep', 'long', 'process', 'markets', 'within', 'due', 'steve', 'staff', 'purchase', '(713)', 'members', 'without', 'meet', 'times', 'intended', 'generation', 'believe', 'company', 'capacity', 'says', 'department', 'scott', 'research', 'continue', 'tax', 'data', 'communications@enron', 'generators', 'web', 'e', 'exchange', '(email)"', 'industry', 'sell', 'position', '', 'say', 'notice', 'tom', 'team', 'option', 'transactions', 'products', 'interest', 'transaction', 'top', 'iso', 'round', 'vice', 'robert', 'mary', 'days', 'site', 'place', 'houston', 'receive', 'years', "enron's", 'eol', 'counterparty', 'needs', 'stock', 'amount', 'increase', 'documents', 'communications', 'direct', 'told', 'program', 'international', 'schedule', 'deals', 'note', 'investment', 'july', 'currently', 'press', 'york', 'sales', 'joe', 'governor', 'senior', 'oil', 'great', 'chief', 'director', 'phone', 'point', 'fax', 'smith', 'bush', 'officials', 'using', 'friday', 'looking', 'law', '', 'chairman', 'around', 'given', 'money', 'terms', '10', 'case', 'member', 'kate', 'marketing', 'getting', 'month', 'policy', 'karen', 'summer', 'got', 'steven', 'southern', 'crisis', '(fax)', '"the', 'either', 'section', 'language', 'e', 'recent', 'second', 'prior', 'line', 'national', 'january', 'possible', 'total', 'term', 'december', 'unit', 'done', 'rights', 'copyright', 'check', 'swap', 'less', 'corporate', 'soon', 'london', 'offer', 'expected', 'november', 'bank', 'called', 'agreements', 'plans', 'approval', 'resources', '1', 'monday', 'version', 'value', 'communications', 'tuesday', 'early', 'talk', 'provided', 'feel', 'scheduled', 'approved', 'local', 'monday', 'full', 'major', 'file', 'pacific', 'senate', 'really', 'account', 'according', 'wednesday', 'required', 'little', 'via', 'house', 'big', 'week', 'clearing', 'venture', 'try', 'hi', '2', 'commercial', 'likely', 'also', 'states', 'r', 'daily', 'move', 'agreed', 'better', 'water', 'jim', 'confirm', 'future', 'least', 'update', 'document', '5', 'weather', '1400', 'pm', 'control', 'thursday', 'finance', 'response', "i've", 'decision', 'anything', 'existing', 'trying', 'subject', 'confidential', 'ceo', 'close', 'october', 'effective', 'stephanie', 'past', 'action', 'futures', 'signed', 'real', 'bob', 'city', 'problem', 'months', 'proposal', 'nymex', 'interested', 'wanted', 'independent', 'friday', 'probably', 'delivery', 'something', 'key', 'los', 'event', 'lisa', 'revised', 'later', 'start', 'retail', 'options', 'party', 'net', 'large', 'individual', 'street', 'fax', 'far', 'thought', 'allow', 'payment', '4', '/', 'making', 'year', 'network', 'immediately', 'return', 'included', 'original', '(phone)', 'ge', 'every', 'add', 'taking', 'problems', 'puc', 'higher', 'respond', 'francisco', 'specific', 'important', 'west', 'bankruptcy', 'eb', 'political', 'deregulation', 'consumers', 'short', 'potential', 'shackleton', 'limited', 'run', 'investors', 'fund', 'copies', 'enough', 'systems', '@', 'distribution', 'average', 'hearing', 'home', "company's", 'street', 'filed', 'software', 'file', 'things', 'visit', 'opportunity', 'funding', 'private', 'although', 'fw', 'announced', '77002', 'related', 'cash', 'co', '(see', 'dear', 'small', 'ask', 'world', 'agreement', 'carol', 'series', 'best', 'understand', 'left', 'different', 'february', 'ken', 'sign', 'gov', 'fuel', 'together', 'provides', 'associated', '"we', 'upon', 'market', 'paid', '09', 'longterm', 'needed', 'employees', 'said', 'alan', 'entity', 'enrononline', 'wants', 'american', 'require', 'period', 'yet', 'center', 'thursday', 'manager', 'share', 'become', 'raised', 'questions', 'former', 'gray', 'william', '', 'angeles', 'time', 'bring', 'someone', 'administration', 'standard', 'held', 'person', 'executed', 'kevin', 'production', 'presentation', 'added', 'businesses', 'four', 'latest', 'tomorrow', 'earlier', 'recently', 'funds', "that's", 'parties', 'sale', 'university', '01)', 'among', 'page', 'shares', 'largest', 'bills', 'peter', 'services', 'court', 'th', 'outside', 'morning', 'result', 'details', 'type', 'today', 'paper', "we're", 'b', 'view', 'know', 'cannot', 'duke', 'complete', 'read', 'certain', 'economic', 'discussed', 'year', 'delete', 'never', 'europe', 'settlement', 'ect', 'reports', 'el', 'show', 'third', 'trades', 'named', 'counsel', 'physical', 'environmental', 'and/or', 'electronic', 'five', 'western', 'question', 'operations', 'commodity', 'join', 'greg', 'cut', 'dan', 'changed', 'confirmation', 'operating', 'along', 'build', 'coming', 'special', 'room', 'blackouts', 'pricing', 'agree', 'ben', 'clear', 'provider', '15', 'mw', 'status', 'rather', 'equity', 'diego', 'dasovich', 'lot', 'united', 'analysis', 'ensure', 'storage', '30', 'area', 'responsible', 'september', 'create', 'submitted', 'spot', 'buying', 'names', 'found', 'rules', 'air', 'necessary', 'school', 'ability', 'notify', 'consumer', 'lower', 'building', 'mail', 'anyone', 'communication', 'link', 'hard', 'hours', 'authority', 'sarashackleton@enroncom', 'gets', 'late', 'washington', 'previous', 'date', '(c)', '', 'agency', 'directly', 'requested', 'focus', 'looks', 'volume', 'respect', 'phone', 'frank', 'yesterday', 'points', 'floor', 'wednesday', 'level', 'regulators', 'significant', 'information', 'corporation', 'pipeline', 'south', 'earnings', 'release', '1', 'dave', 'managing', 'reference', 'tuesday', 'act', 'begin', 'job', 'broadband', 'seems', 'questions', 'issued', 'summary', 'happy', 'charge', 'california', 'assets', 'grid', '(email);', 'cover', 'everyone', 'efforts', 'lay', 'enron', 'foreign', 'brent', 'taylor/hou/ect@ect', 'white', 'partners', 'reserved', 'develop', '6', 'time', 'similar', 'consider', 'discussion', 'august', 'others', 'follow', 'behalf', 'legislation', 'cap', 'appropriate', 'derivatives', 'jeffrey', 'competitive', 'week', '7136463490', 'confidentiality', 'india', 'across', 'filing', 'heard', 'hold', 'taken', 'reduce', '(the', 'strong', 'lead', 'group', 'though', 'experience', 'w', 'thanks!', 'leslie', 'load', 'tim', 'n', 'jones/hou/ect@ect', 'came', 'caps', "(email)';", 'well', 'tell', 'construction', 'facility', 'application', 'industrial', 'division', 'initial', 'group', 'weeks', 'nothing', 'works', 'dabhol', 'suggested', 'sheila', 'expect', 'numbers', 'growth', 'basis', 'llc', 'model', 'traders', 'users', '[image]', 'various', 'california', '', 'role', 'george', 'shackleton/hou/ect@ect', 'enter', 'authorized', 'tanya', 'performance', 'affairs', 'remain', 'kind', 'appreciate', 'sold', 'error', 'enron', 'impact', '25', 'saying', 'includes', 'seen', 'means', 'shackleton/hou/ect', 'turn', 'spokesman', 'h', 'company', 'sending', 'makes', 'increased', 'firms', 'sorry', 'debt', 'idea', 'unless', 'partners', 'etc', 'pass', 'talking', 'european', 'providing', 'involved', 'peak', 'comment', 'media', 'index', 'contain', 'hear', 'fact', 'revenue', 'advise', 'transfer', 'securities', 'projects', 'security', 'force', 'assignment', 'ready', '"i', 'fyi', 'addition', 'structure', 'completed', 'hour', 'telephone', 'leave', 'system', 'assembly', 'positions']
#print(top_3)
#punctuation = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'",'____','=']
filtered_words = [word for word in top_3 if word not in stopwords.words('english')]
print(filtered_words)
kickoff = [item.replace("()", "") for item in top_3]
# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = """To l 3 ca  "'' 01 00 (email)\ 7 de c g  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (713)[ ] I you am As it  can't  <<...>>  sincerely, .  > - < Kenneth Lay/Corp/Enron@Enron Best regards Sincerely From  Sent Original Message Q <-> * j  i'll |  \ /\ 100% 12345678910  (email)"' () """
# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords.split()
clean = [word for word in kickoff if word not in stoplist]
document=clean
#print(document)


tfidf_vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')

dtm = tfidf_vectorizer.fit_transform(document)  # sparse matrix with columns corresponding to words
tfidf_vectorizer.get_feature_names()

# Apply the vectoriser to the training set
Cardinality=0
for files in document:
    if files.endswith('.txt'):
        Cardinality+=1
counts = CountVectorizer(input='document')
dtm = counts.fit_transform(document)  # a sparse matrix
vocab = counts.get_feature_names()  # a list
#type(dtm)
dtm = dtm.toarray()  # convert to a regular array
#print (dtm.shape)
N, K = dtm.shape
ind = np.arange(N)  # points on the x-axis
width = 0.2
vocab = np.array(vocab)
n, _ = dtm.shape
dist = np.zeros((n, n))

#dissimilarity
Dissimilarity=dist
for i in range(n):
     for j in range(n):
        x, y = dtm[i, :], dtm[j, :]
        dist[i, j] = np.sqrt(np.sum((x - y)**2))
matrix = tfidf_vectorizer.fit_transform(document)


vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

text = document


for token in text:
    syn_sets = wn.synsets(token)
    for syn_set in syn_sets:
        print(syn_set, syn_set.lemma_names())
        print(syn_set.hyponyms())
        print(syn_set.definition())
        #similarities = [syn_set.path_similarity(t)*10 for t in syn_sets]

        #print (syn_set.name.partition('.')[0])

sims = []

for word1, word2 in product(top_3, text):
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)
    for sense1, sense2 in product(syns1, syns2):
        d = wn.wup_similarity(sense1, sense2)
        sims.append((d, syns1, syns2))

allsyns1 = set(ss for word in top_3 for ss in wn.synsets(word))
allsyns2 = set(ss for word in text for ss in wn.synsets(word))
best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in
        product(allsyns1, allsyns2))
best2 = max((wn.path_similarity(s1, s2) or 0, s1, s2) for s1, s2 in
        product(allsyns1, allsyns2))
best3 = max((wn.lch_similarity(s1, s2) or 0, s1, s2) for s1, s2 in
        product(allsyns1, allsyns2))

print(best, best2, best3)




