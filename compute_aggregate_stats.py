import argparse
import json
import time
from collections import defaultdict
import numpy as np

def main(params):
    res= json.load(open(params['inputCands'],'r'))
    icnt = 0

    if 'misc' not in res:
        if params['age']:
            auth_to_ix = {'<50':0,'<20':1}
            ix_to_auth = {0:'<50',1:'<20'}
        else:
            auth_to_ix = {'male':0,'female':1}
            ix_to_auth = {0:'male',1:'female'}
    else:
        auth_to_ix = res['misc']['auth_to_ix']
        ix_to_auth = res['misc']['ix_to_auth']
        auth_to_ix = {k:int(auth_to_ix[k]) for k in auth_to_ix}
        ix_to_auth = {int(k):ix_to_auth[k] for k in ix_to_auth}

    #for doc in res['docs']:
    #    ix = auth_to_ix[doc['author']]
    #    for st in doc['sents']:
    #        inpset = set(st['sent'].split()[:-1])
    #        if len(inpset) > 0 and st['score'][1-ix]>params['filter']:
    #            genset = set(st['trans'].split()[:-1])
    #            recall[ix].append(float(len(inpset & genset))/float(len(inpset)))
    #            sents[ix].append(st['sent'])
    #            sc[ix].append(st['score'][1-ix])
    #            trans_sents[ix].append(st['trans'])
    #            diff_sc[ix].append(st['trans_score'][ix] - st['score'][ix])
    #            if params['semantic_embedding']:
    #                emb_diff[ix].append(np.abs(sem_emb[st['trans_enc'],:] - sem_emb[st['sent_enc'],:]).sum())


    doc_accuracy = np.zeros(len(auth_to_ix))
    doc_accuracy_trans = np.zeros(len(auth_to_ix))
    doc_count = np.zeros(len(auth_to_ix))
    for doc in res['docs']:
        doc_score_orig = np.array([0.,0.])
        doc_score_trans = np.array([0.,0.])
        for st in doc['sents']:
            if type(st) == list:
                all_m_scr = np.array([sent[params['max']] for sent in st])
                m_idx = all_m_scr.argmax()
                doc_score_orig  += np.log(st[m_idx]['score'])
                doc_score_trans += np.log(st[m_idx]['trans_score'])
            else:
                doc_score_orig  += np.log(st['score'])
                doc_score_trans += np.log(st['trans_score'])
        doc_accuracy[auth_to_ix[doc['author']]] += float(doc_score_orig.argmax() == auth_to_ix[doc['author']])
        doc_accuracy_trans[auth_to_ix[doc['author']]] += float(doc_score_trans.argmax() == auth_to_ix[doc['author']])
        doc_count[auth_to_ix[doc['author']]] += 1.

    print 'Original data'
    print '-------------'
    print 'Doc accuracy is %s : %.2f , %s : %.2f'%(ix_to_auth[0], (doc_accuracy[0]/doc_count[0]),ix_to_auth[1], (doc_accuracy[1]/doc_count[1]) )
    fp = doc_count[1]- doc_accuracy[1]
    recall = doc_accuracy[0]/doc_count[0]
    precision = doc_accuracy[0]/(doc_accuracy[0]+fp)
    f1score = 2.*(precision*recall)/(precision+recall)
    print 'Precision is %.2f : Recall is %.2f , F1-score is %.2f'%(precision, recall, f1score)
    print '\nTranslated data'
    print '-----------------'
    print 'Doc accuracy is %s : %.2f , %s : %.2f'%(ix_to_auth[0], (doc_accuracy_trans[0]/doc_count[0]),ix_to_auth[1], (doc_accuracy_trans[1]/doc_count[1]) )
    fp = doc_count[1]- doc_accuracy_trans[1]
    recall = doc_accuracy_trans[0]/doc_count[0]
    precision = doc_accuracy_trans[0]/(doc_accuracy_trans[0]+fp)
    f1score = 2.*(precision*recall)/(precision+recall)
    print 'Precision is %.2f : Recall is %.2f , F1-score is %.2f'%(precision, recall, f1score)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('inputCands', type=str, help='the input candidateJson')
  parser.add_argument('--max', type=str, default='meteor',help='the input candidateJson')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)

