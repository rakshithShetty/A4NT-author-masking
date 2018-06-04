import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import json
import argparse
import re
from collections import defaultdict
import csv
def cleantext(sent):
    return re.sub('END','',re.sub('ELIP','...',re.sub(r'3(\S)',r'\1\1\1',sent)))

def get_sentence(st, metric='METEOR', method='max', m_idx = None):
    if m_idx == None:
        all_m_scr = np.array([sent[metric] for sent in st])
        if method == 'max':
            m_idx = all_m_scr.argmax()
        elif method == 'min':
            m_idx = all_m_scr.argmin()
        elif method == 'rand':
            m_idx = np.random.choice(len(all_m_scr),1)
        else:
            m_idx = int(method)
    st = st[m_idx]
    return st,m_idx

def main(params):
  ores = []
  trans_sents = []
  for oresf in params['reslist']:
      ores.append(json.load(open(oresf,'r')))
      trans_sents.append([])
      totalsent = 0
      for i,doc in enumerate(ores[-1]['docs']):
          ores[-1]['docs'][i]['dict_sents'] = {}
          for j, st in enumerate(doc['sents']):
              totalsent +=1
              if type(st) == list:
                  ores[-1]['docs'][i]['dict_sents'][st[0]['sid']] = st
              else:
                  ores[-1]['docs'][i]['dict_sents'][st['sid']] = st

  docid = []
  sid = []
  sents = []
  for i,doc in enumerate(ores[-1]['docs']):
      for j,st in enumerate(doc['sents']):
          sents.append(st[0]['sent'])
          docid.append(doc['id'])
          sid.append(st[0]['sid'])
          for oi in xrange(len(ores)):
              ost, _ = get_sentence(ores[oi]['docs'][i]['dict_sents'][st[0]['sid']],method=params['filter_by'][oi])
              trans_sents[oi].append(ost['trans'])


  outf = open(params['outfile'],'w')
  outwriter = csv.writer(outf)
  outwriter.writerow(['reference']+['sent'+str(i+1) for i in xrange(len(ores))] + ['name'+str(i+1) for i in xrange(len(ores))]+['docid','sid'])
  count =0
  maxcount = min(params['ndump'],totalsent) if params['ndump']>0 else totalsent
  all_ids =  np.arange(len(sents))
  np.random.shuffle(all_ids)
  total_skip = 0
  skip_wins=defaultdict(int)
  for idx in all_ids[:maxcount]:
      cur_order = np.random.permutation(len(ores))
      if params['skip_duplicates']:
          ref = cleantext(sents[idx]).strip()
          skip = False
          for ci in cur_order:
              if cleantext(trans_sents[ci][idx]).strip() == ref:
                  skip_wins[params['names'][ci]] += 1
                  total_skip += 1
                  skip = True
                  break
          if skip:
              continue

      outwriter.writerow([cleantext(sents[idx]).strip()]+[cleantext(trans_sents[ci][idx]).strip() for ci in cur_order]+[params['names'][ci] for ci in cur_order]+[docid[idx],sid[idx]] )
  if params['skip_duplicates']:
      print 'Total skips %d'%total_skip
      for k in params['names']:
          print '%s : %d'%(k,skip_wins[k])
  outf.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-o','--outfile', dest='outfile', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-n','--ndump', dest='ndump', type=int, default=-1, help='batch_size to use')
  parser.add_argument('--reslist', dest='reslist', type=str, nargs='+', default=[], help='generator/GAN checkpoint filename')
  parser.add_argument('--names', type=str, nargs='+',default=[],help='the input candidateJson')
  parser.add_argument('--filter_by', type=str, nargs='+',default=[],help='the input candidateJson')
  parser.add_argument('-s','--shufflecols', type=int, default=1, help='generator/GAN checkpoint filename')
  parser.add_argument('-d','--skip_duplicates', type=int, default=0, help='generator/GAN checkpoint filename')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
