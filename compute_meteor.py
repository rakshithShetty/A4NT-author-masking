import argparse
import json
import time
from collections import defaultdict
import numpy as np
import cPickle as pickle
from eval.mseval.pycocoevalcap.meteor.meteor import Meteor
from eval.mseval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from eval.mseval.pycocoevalcap.bleu.bleu import Bleu
from eval.mseval.pycocoevalcap.rouge.rouge import Rouge
from eval.mseval.pycocoevalcap.cider.cider import Cider


def main(params):
    resInp = json.load(open(params['inputCands'],'r'))
    resGtImgid = defaultdict(list)
    resCandsImgid = defaultdict(list)
    icnt = 0
    for i,doc in enumerate(resInp['docs']):
        imgid = str(i)
        for j,st in enumerate(doc['sents']):
            if type(st)==list:
                for sent in st:
                    resCandsImgid[imgid+'+'+str(j)].append({'image_id':imgid,'caption':' '.join(sent['trans'].split()[:-1]),'id':icnt})
                    resGtImgid[imgid+'+'+str(j)].append({'image_id':imgid,'caption':' '.join(sent['sent'].split()[:-1]),'id':icnt})
                    icnt+=1
            else:
                resCandsImgid[imgid+'+'+str(j)].append({'image_id':imgid,'caption':' '.join(st['trans'].split()[:-1]),'id':icnt})
                resGtImgid[imgid+'+'+str(j)].append({'image_id':imgid,'caption':' '.join(st['sent'].split()[:-1]),'id':icnt})
                icnt+=1
    tokenizer = PTBTokenizer()
    resCandsImgid = tokenizer.tokenize(resCandsImgid)
    resGtImgid = tokenizer.tokenize(resGtImgid)

    eval_metric = params['eval_metric']
    if eval_metric == 'meteor':
      scorer = Meteor()
      scorer_name = "METEOR"
    elif eval_metric == 'cider':
      scorer = Cider()
      scorer_name = "CIDEr"
    elif eval_metric == 'rouge':
      scorer = Rouge()
      scorer_name = "ROUGE_L"
    elif eval_metric[:4] == 'bleu':
      bn = int(eval_metric.split('_')[1])
      scorer = Bleu(bn)
      scorer_name = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
      scorer_name = scorer_name[:bn]
    else:
      raise ValueError('ERROR: %s --> Unsupported eval metric'%(eval_metric))

    lenDict = defaultdict(list)
    for k in resCandsImgid:
       lenDict[len(resCandsImgid[k])].append(k)

    maxlen = max(lenDict.keys())
    print 'Max length: %d'%maxlen
    for i in xrange(maxlen):
        res ={}
        gts = {}
        for k in resGtImgid.keys():
            if i < len(resCandsImgid[k]):
                res[k] = [resCandsImgid[k][i]]
                gts[k] = resGtImgid[k]
        print 'Now in %d, Lengths %d'%(i, len(gts))
        t0 = time.time()
        score, scores = scorer.compute_score(gts, res)
        dt = time.time() - t0
        print 'Done %d in %.3fs, score = %.3f' %(i, dt, score)
        icnt = 0
        for si,k in enumerate(gts.keys()):
            idx,sidx = map(int,k.split('+'))
            if type(st)==list:
                resInp['docs'][idx]['sents'][sidx][i][scorer_name] = scores[si]
            else:
                resInp['docs'][idx]['sents'][sidx][scorer_name] = scores[si]

        assert(len(scores) == si+1)
    #pickle.dump(candScoresImgid,open('candScrMeteor_4AuxCmmePgoogSwapPposJJ_fullVal.json','w'))
    json.dump(resInp,open(params['inputCands'],'w'))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('inputCands', type=str, help='the input candidateJson')
  parser.add_argument('-m',dest='eval_metric', type=str, default='meteor', help='Which metric to use for eval')

  parser.add_argument('-r', dest='refdata', type=str, default='/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/captions_val2014.json', help='file with reference captions')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
