import argparse
import csv
from collections import defaultdict
import numpy as np

def main(params):
  res = open(params['resultfile'],'r')
  resreader = csv.reader(res)
  resDict = defaultdict(int) 
  resDictCombo = defaultdict(int)
  cols = resreader.next()
  ans_col = cols.index('Answer.sentiment')

  modelCols = [cols.index('Input.name'+str(i+1)) for i in xrange(params['num_models'])]
  docid_col = cols.index('Input.docid')
  sid_col = cols.index('Input.sid')
  ref_id = cols.index('Input.reference')
  sent_set = set()
  totalCount = 0.
  majority_votes = 0
  votes_per_entry = defaultdict(lambda: defaultdict(int))
  for row in resreader:
      sid = int(row[sid_col])
      docid = int(row[docid_col])
      votes_per_entry[(docid,sid)][row[modelCols[int(row[ans_col][-1])-1]]] += 1
      sent_set.add(row[ref_id].strip())

  totalCount = len(votes_per_entry)
  models = [row[mid] for mid in modelCols]
  print 'Models are ', models 
  for k in votes_per_entry:
      all_counts = np.array([votes_per_entry[k][mid] if mid in votes_per_entry[k] else 0 for mid in models])
      max_count = all_counts.max()
      #Check for uniqueness of maximum
      if (all_counts==max_count).sum() == 1:# and (params['majority_vote'] ==1):
        resDict[models[all_counts.argmax()]] += 1
        majority_votes += 1
      else:
          print all_counts
      #elif params['majority_vote'] == 0:
      #  for i in xrange(len(all_counts)):
      #      resDict[models[all_counts.argmax()]] += all_counts[i]
      #  majority_votes += 1


  print 'Total Entries = %d, Majority count = %d '%(totalCount, majority_votes)
  print 'Final Counts are :'
  for k in resDict:
    print'%s = %.2f, %d'%(k, 100. * float(resDict[k])/totalCount, resDict[k])
    print'%s = %.2f, %d'%(k, 100. * float(resDict[k])/totalCount, resDict[k])
  print '--------------------------------------------'

  if params['skipped_data']:
    skipped = open(params['skipped_data'],'r')
    skipreader = csv.reader(skipped)

    cols = skipreader.next()
    skip_modelCols = [cols.index('name'+str(i+1)) for i in xrange(params['num_models'])]
    skip_sentCols = [cols.index('sent'+str(i+1)) for i in xrange(params['num_models'])]
    skip_docid_col = cols.index('docid')
    skip_sid_col = cols.index('sid')
    skip_ref_id = cols.index('reference')
    
    skip_resDict = defaultdict(int) 
    skip_resDict_combos = defaultdict(int)
    tot_skipped = 0
    tot_skip_wins = 0
    tot_bad_skips = 0


    for row in skipreader:
        ref = row[skip_ref_id].strip()
        matched_cols = []
        if ref not in sent_set:
            tot_skipped += 1
            for i in xrange(params['num_models']):
                if row[skip_sentCols[i]].strip() == ref:
                    matched_cols.append(row[skip_modelCols[i]].strip())
            if len(matched_cols) == 1:
                skip_resDict[matched_cols[0]] += 1
                skip_resDict_combos[matched_cols[0]] += 1
                tot_skip_wins +=1
            elif len(matched_cols) > 1 and params['majority_vote']==0:
                for i in xrange(len(matched_cols)):
                    skip_resDict[matched_cols[i]] += 1
                skip_resDict_combos['+'.join([mc for mc in matched_cols])] += 1
                tot_skip_wins +=len(matched_cols)
                #tot_skipped +=(len(matched_cols) - 1)
            elif len(matched_cols) == 0:
                tot_bad_skips +=1

    tot_skipped = tot_skipped - tot_bad_skips

    print '--------------------------------------------'
    print 'Total Skip Entries = %d, Skip majority count = %d '%(tot_skipped, tot_skip_wins)
    print 'Total Bad Skip = %di'%(tot_bad_skips)
    print 'Final Counts are :'
    for k in resDict:
      print'%s = %.2f, %d'%(k, 100. * float(skip_resDict[k] + resDict[k])/(totalCount+tot_skipped), skip_resDict[k])
    
    print 'Pecentage of ties = %.2f, total = %d'%(100.*(1.-float(tot_skip_wins + majority_votes)/(totalCount+tot_skipped)), totalCount+tot_skipped-tot_skip_wins - majority_votes)
    print skip_resDict_combos



    


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(dest='resultfile', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-m','--num_models', dest='num_models', type=int, default=4, help='how many characters to generate per string')
  parser.add_argument('--majority_vote', dest='majority_vote', type=int, default=1, help='how many characters to generate per string')
  parser.add_argument('--include_skipped', dest='skipped_data', type=str, default=None, help='how many characters to generate per string')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
