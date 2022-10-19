'''
Author: Paul Tomoiaga ( pault@tomopaul.com )
Created for Web Analytics Wednesday : 2022-10-19
Example dataset: https://huggingface.co/datasets/app_reviews

Description:
Program aids in manual topic modelling by automating the process of
finding common, meaningful keywords. It works by counting the top-most common words,
and filtering unwanted words using an exhaustive list of stopwords.

Output is a list of files; each file a subset of the original feedback log
with records corresponding to a particular word.
i.e. file "00_app_1234.csv" records all feedback corresponding to the word "app".

A user should then manually analyse each file for 
'''

import pandas as pd
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopword_tokens
import os

#--- Read Dataset ---
#--------------------
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().split('\n'))
stopwords = stopwords.union({'telegram', 'whatsapp', 'nice', 'app', 'super', 'awesome', 'love'})

raw = pd.read_csv('telegram_reviews.csv')
raw_clean = raw.dropna(subset=['review'])

#--- Remove Stopwords ---
#------------------------
corpus = [remove_stopword_tokens(simple_preprocess(doc, deacc=True), stopwords=stopwords)
          for doc in raw_clean['review']]

try:
    os.mkdir('output')
except FileExistsError:
    pass

#--- Generate Output ---
#-----------------------
dct = Dictionary(corpus) # Dictionary object already counts word frequency

for i, word in enumerate(dct.most_common(50)):
    # 00_app_1234.csv
    filename = 'output/' + str(i).zfill(2) + '_' + word[0] + '_' + str(word[1]) + '.csv'

    result = raw_clean[raw_clean['review'].str.contains(word[0])]
    result.to_csv(filename, index=False)
