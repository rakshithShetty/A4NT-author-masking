import numpy as np
import json
import argparse

def main(params):
    resf = params['resfile']
    res = json.load(open(resf,'r'))
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

    sents = [[],[]]
    trans_sents = [[],[]]
    diff_sc = [[],[]]
    recall = [[],[]]
    emb_diff = [[],[]]
    if params['semantic_embedding']:
        sem_emb = np.load(params['semantic_embedding'])

    for doc in res['docs']:
        ix = auth_to_ix[doc['author']]
        for st in doc['sents']:
            inpset = set(st['sent'].split()[:-1])
            if len(inpset) > 0:
                genset = set(st['trans'].split()[:-1])
                recall[ix].append(float(len(inpset & genset))/float(len(inpset)))
                sents[ix].append(st['sent'])
                trans_sents[ix].append(st['trans'])
                diff_sc[ix].append(st['trans_score'][ix] - st['score'][ix])
                if params['semantic_embedding']: 
                    emb_diff[ix].append(np.abs(sem_emb[st['trans_enc'],:] - sem_emb[st['sent_enc'],:]).sum())

    diff_sc = [np.array(diff_sc[0]), np.array(diff_sc[1])]
    recall = [np.array(rc) for rc in recall]
    if params['semantic_embedding']: 
        emb_diff = [np.array(emb) for emb in emb_diff]
    for j in xrange(2):
        print '\n----------------------------------------------------------'
        print 'Author %s to %s translation'%(ix_to_auth[j], ix_to_auth[1-j])
        print '----------------------------------------------------------'
        score = diff_sc[j] * recall[j]
        for i in (score).argsort()[params['offset']:params['offset']+params['ndisp']]:
            if params['semantic_embedding']: 
                print 'diff %.2f, recall: %.2f, emb_diff: %.2f'%(-diff_sc[j][i],recall[j][i], emb_diff[j][i])
            else:
                print 'diff %.2f, recall: %.2f'%(-diff_sc[j][i],recall[j][i])
            print 'Inp %6s --> %s'%(ix_to_auth[j],sents[j][i])
            print 'Out %6s --> %s'%(ix_to_auth[1-j],trans_sents[j][i])
            print ''


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-r','--resfile', dest='resfile', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-n','--ndisp', dest='ndisp', type=int, default=10, help='batch_size to use')
  parser.add_argument('-a','--age', dest='age', type=int, default=1, help='batch_size to use')
  parser.add_argument('-o','--offset', dest='offset', type=int, default=0, help='batch_size to use')
  parser.add_argument('-s','--semantic_embedding', dest='semantic_embedding', type=str, default=None, help='batch_size to use')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
