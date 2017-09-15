import numpy as np
import json
import argparse

def main(params):
    resf = params['resfile']
    res = json.load(open(resf,'r'))
    if params['age']:
        auth_to_ix = {'<50':0,'<20':1}
        ix_to_auth = {0:'<50',1:'<20'}
    else:
        auth_to_ix = {'male':0,'female':1}
        ix_to_auth = {0:'male',1:'female'}

    sents = [[],[]]
    trans_sents = [[],[]]
    diff_sc = [[],[]]
    recall = [[],[]]

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

    diff_sc = [np.array(diff_sc[0]), np.array(diff_sc[1])]
    recall = [np.array(rc) for rc in recall]

    for j in xrange(2):
        print '\n----------------------------------------------------------'
        print 'Author %s to %s translation'%(ix_to_auth[j], ix_to_auth[1-j])
        print '----------------------------------------------------------'
        for i in (diff_sc[j]*recall[j]).argsort()[params['offset']:params['offset']+params['ndisp']]:
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


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
