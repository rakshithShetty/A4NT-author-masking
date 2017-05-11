import os
import os.path as osp
import numpy as np
import unicodedata
import json
import chardet
import io
from collections import Counter
import re

def preproc_dataset(src_path, output_path, frac_test = 0.15, frac_val = 0.1):
    authors = {}
    frac_test = 0.15
    frac_val = 0.1
    src_path = 'data/pan16AuthorMask/splits/train/'

    for ath in os.listdir(src_path):
        authors[ath] = {'src_files':[]}
        authors[ath]['src_files'] = os.listdir(osp.join(src_path,ath))

    # Create the outer structure holding all the data
    data = {'author-data':authors, 'text-stats':{}}

    # Compute some statistics for the dataset
    #---------------------------------------------------------------------------
    per_auth_docs = np.array([len(authors[auth]['src_files']) for auth in authors])
    data['text-stats']['total-docs'] = per_auth_docs.sum()
    data['text-stats']['max-docs'] = per_auth_docs.max()
    data['text-stats']['min-docs'] = per_auth_docs.min()
    data['text-stats']['median-docs'] = np.median(per_auth_docs)
    n_total = per_auth_docs.sum()
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Divide the data into train/ val/ and test splits
    # Constraints chose are: atleast one training document left for each author
    # Ideally the original.txt could be used as val or test data, as that is the
    # target for obfuscation.
    #---------------------------------------------------------------------------
    rem_author_docs = per_auth_docs - 1
    val_idx = []; test_idx = []
    n_val = 0; n_test = 0;
    #-------------------------Decide which authors contribute to test--------------
    while (n_test < (n_total*frac_test)//1):
        idx = np.random.randint(0,per_auth_docs.shape[0],1)
        if rem_author_docs[idx] > 0:
            rem_author_docs[idx] = rem_author_docs[idx] - 1
            test_idx.append(idx[0])
            n_test = len(test_idx)

    #-------------------------Decide which authors contribute to val--------------
    while (n_val < (n_total*frac_val)//1):
        idx = np.random.randint(0,per_auth_docs.shape[0],1)
        if rem_author_docs[idx] > 0:
            rem_author_docs[idx] = rem_author_docs[idx] - 1
            val_idx.append(idx[0])
            n_val = len(val_idx)

    #------------------------------Initialize all as train first--------------------
    for i,auth in enumerate(authors):
        authors[auth]['split'] = ['train']*len(authors[auth]['src_files'])

    #------------------------------Assign docs to test split------------------------
    for i in test_idx:
        aid = authors.keys()[i]
        oid = authors[aid]['src_files'].index('original.txt')
        if authors[aid]['split'][oid] == 'train':
            authors[aid]['split'][oid] = 'test'
            continue
        else:
            done = 0
            while not done:
                idx = np.random.randint(0, len(authors[aid]['split']),1)
                if authors[aid]['split'][idx[0]] == 'train':
                    authors[aid]['split'][idx[0]] = 'test'
                    done = 1
    #-------------------------------Assign docs to val split------------------------
    for i in val_idx:
        aid = authors.keys()[i]
        oid = authors[aid]['src_files'].index('original.txt')
        if authors[aid]['split'][oid] == 'train':
            authors[aid]['split'][oid] = 'val'
            continue
        else:
            done = 0
            while not done:
                idx = np.random.randint(0, len(authors[aid]['split']),1)
                if authors[aid]['split'][idx[0]] == 'train':
                    authors[aid]['split'][idx[0]] = 'val'
                    done = 1
    data['author-data'] = authors
    #---------------------------------------------------------------------------

    all_docs = []
    for aid in authors:
        for i,sf in enumerate(authors[aid]['src_files']):
            all_docs.append({'author':aid, 'src': sf, 'split': authors[aid]['split'][i]})

    data['docs'] = all_docs

    #---------------------------Read text data and normalize it------------------
    for i, dc in enumerate(data['docs']):
        filename = osp.join(src_path,dc['author'],dc['src'])
        encoding = 'utf-8-sig'
        with io.open(filename, encoding=encoding) as file:
          text = file.read()
        text = unicodedata.normalize('NFKD', text)
        data['docs'][i]['text'] = text
    #---------------------------------------------------------------------------
    data['docs'] = normalize_text(data['docs'])


    json.dump(data, open(osp.join(output_path, 'dataset.json'),'w'))

    return data


def normalize_text(docs):
    for i,dc in enumerate(docs):
        txt = dc['text']
        # Remove all \n
        txt = re.sub('\\n', '',txt)
        # First replace all numbers with 0
        txt = re.sub('\d', '0',txt)
        #Split capital letters to <marker><lower> form using 1 as the marker.
        # Eg. A --> 1a
        txt = re.sub(r'([A-Z])',lambda pat: '1'+pat.group(1).lower(), txt)
        # Use 2 to indicate beginning of the document and 3 to indicate end of doc
        docs[i]['text'] = '2'+txt+'3'
    return docs

