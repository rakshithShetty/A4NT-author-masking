import os
import os.path as osp
import numpy as np
import unicodedata
import json
import chardet
import io
from collections import Counter

authors = {}

for ath in os.listdir('.'):
    authors[ath] = {'src_files':[]}
    authors[ath]['src_files'] = os.listdir(osp.join('.',ath))

data = {'author-data':authors, 'text-stats':{}}
per_auth_docs = np.array([len(authors[auth]['src_files']) for auth in authors])
data['text-stats']['total-docs'] = per_auth_docs.sum()
data['text-stats']['max-docs'] = per_auth_docs.max()
data['text-stats']['min-docs'] = per_auth_docs.min()
data['text-stats']['median-docs'] = np.median(per_auth_docs)

vocab = Counter()
for auth in authors:
    authors[auth]['text-data'] = []
    for i,fname in enumerate(authors[auth]['src_files']):
      filename = osp.join(auth,fname)
      #with open(filename,'rb') as file:
      #  raw = file.read(32)
      #  encoding = chardet.detect(raw)['encoding']
      encoding = 'utf-8-sig'
      with io.open(osp.join(auth,'original.txt'), encoding=encoding) as file:
        text = file.read()
      text = unicodedata.normalize('NFKD', text)
      vocab.update([c for c in text])
      authors[auth]['text-data'].append(text)
data['author-data'] = authors
json.dump(data, open('../dataset.json','w'))
