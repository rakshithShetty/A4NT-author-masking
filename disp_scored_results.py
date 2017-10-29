import numpy as np
import json
import argparse
from termcolor import colored

def color_diffs(inp,ref, colors=[None,'on_green'], attrs = []):
    ref_tkns = set(ref.split())
    out = ' '.join([colored(w, 'grey',colors[int(w not in set(ref.split()))],attrs=attrs) for w in inp.split()])
    return out

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
            m_idx = int(metric)
    st = st[m_idx]
    return st,m_idx

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

    if params['otherres']:
        ores = []
        other_trans_sents = []
        for oresf in params['otherres']:
            ores.append(json.load(open(oresf,'r')))
            other_trans_sents.append([[],[]])
            for i,doc in enumerate(ores[-1]['docs']):
                ores[-1]['docs'][i]['dict_sents'] = {}
                for j, st in enumerate(doc['sents']):
                    if type(st) == list:
                        ores[-1]['docs'][i]['dict_sents'][st[0]['sid']] = st
                    else:
                        ores[-1]['docs'][i]['dict_sents'][st['sid']] = st

    sents = [[],[]]
    trans_sents = [[],[]]
    trans_sents_other = [[],[]]
    diff_sc = [[],[]]
    recall = [[],[]]
    emb_diff = [[],[]]
    if params['semantic_embedding']:
        sem_emb = np.load(params['semantic_embedding'])
    sc = [[],[]]
    auth_colors = ['red', 'blue']
    m_idx = 0
    meteor_score = [[],[]]
    for i,doc in enumerate(res['docs']):
        ix = auth_to_ix[doc['author']]
        for j,st in enumerate(doc['sents']):
            if type(st) == list:
                st, m_idx = get_sentence(st,params['filter_score'], params['filter_by'])

            inpset = set(st['sent'].split()[:-1])
            if len(inpset) > 0 and (st['score'][1-ix]>params['filter'][0]) and (st['score'][1-ix]<params['filter'][1]) and (params['lenfilter']==0 or (len(inpset) < params['lenfilter'])):
                genset = set(st['trans'].split()[:-1])
                delta = len(inpset - genset) + len(genset-inpset )
                if params['deltafilter'] < 0 or (delta == params['deltafilter']):
                    recall[ix].append(float(len(inpset & genset))/float(len(inpset)))
                    meteor_score[ix].append(st[params['filter_score']])
                    sents[ix].append(st['sent'])
                    sc[ix].append(st['score'][1-ix])
                    trans_sents[ix].append(st['trans'])
                    if params['show_samples']:
                        trans_sents_other[ix].append([ost['trans'] for osti,ost in enumerate(doc['sents'][j]) if osti!=m_idx])
                    if params['otherres']:
                        for oi in xrange(len(ores)):
                            ost, _ = get_sentence(ores[oi]['docs'][i]['dict_sents'][st['sid']],params['filter_score'], params['other_filter_by'])
                            other_trans_sents[oi][ix].append(ost['trans'])
                    diff_sc[ix].append(st['trans_score'][1-ix] - st['score'][1-ix])
                    if params['semantic_embedding']:
                        emb_diff[ix].append(np.abs(sem_emb[st['trans_enc'],:] - sem_emb[st['sent_enc'],:]).sum())

    diff_sc = [np.array(diff_sc[0]), np.array(diff_sc[1])]
    recall = [np.array(rc) for rc in recall]
    meteor = [np.array(mt) for mt in meteor_score]
    sc = [np.array(s) for s in sc]
    if params['semantic_embedding']:
        emb_diff = [np.array(emb) for emb in emb_diff]
    for j in xrange(2):
        print '\n----------------------------------------------------------'
        print 'Author %s to %s translation'%(ix_to_auth[j], ix_to_auth[1-j])
        print '----------------------------------------------------------'
        #score = diff_sc[j] * meteor[j]
        score = -(((diff_sc[j] + sc[j])>0.5).astype(float) * meteor[j])
        #score = np.random.rand(len(recall[j]))#-(1-recall[j]) * meteor[j]
        #score = diff_sc[j]#-(1-recall[j]) * meteor[j]
        for i in (score).argsort()[params['offset']:params['offset']+params['ndisp']]:
            if params['semantic_embedding']:
                print 'diff %.2f, recall: %.2f, emb_diff: %.2f'%(-diff_sc[j][i],recall[j][i], emb_diff[j][i])
            else:
                print 'diff %.2f, recall: %.2f, meteor: %.2f, orig: %.2f'%(diff_sc[j][i],recall[j][i], meteor[j][i], sc[j][i])
            #print colored('Inp %6s -->'%(ix_to_auth[j]),'green') + colored(' %s'%(sents[j][i]), auth_colors[j])
            #print colored('Out %6s -->'%(ix_to_auth[1-j]),'grey') + colored(' %s'%(trans_sents[j][i]), auth_colors[1-j])
            #if params['otherres']:
            #    print colored('Out %6s -->'%(ix_to_auth[1-j]),'cyan') + colored(' %s'%(other_trans_sents[j][i]), auth_colors[1-j])
            print colored(' Inp %6s --> '%(ix_to_auth[j]),'green') + color_diffs(sents[j][i],trans_sents[j][i], [None,'on_yellow'],attrs=['bold'])
            name = params['model_name'] if params['model_name'] else 'Out'
            print colored('%4s %6s --> '%(name,ix_to_auth[1-j]),'magenta') + color_diffs(trans_sents[j][i],sents[j][i], [None,'on_green'])
            if params['show_samples']:
                for osti in xrange(len(trans_sents_other[j][i])):
                    print colored('%4s %6s --> '%(name,ix_to_auth[1-j]),'magenta') + color_diffs(trans_sents_other[j][i][osti],sents[j][i], [None,'on_green'])
            if params['otherres']:
                for oi in xrange(len(ores)):
                    name = params['other_names'][oi] if params['other_names'] else 'Out'
                    print colored('%4s %6s --> '%(name,ix_to_auth[1-j]),'magenta') + color_diffs(other_trans_sents[oi][j][i],sents[j][i], [None, 'on_green'])
            print ''


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-r','--resfile', dest='resfile', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-n','--ndisp', dest='ndisp', type=int, default=10, help='batch_size to use')
  parser.add_argument('-a','--age', dest='age', type=int, default=1, help='batch_size to use')
  parser.add_argument('-o','--offset', dest='offset', type=int, default=0, help='batch_size to use')
  parser.add_argument('-s','--semantic_embedding', dest='semantic_embedding', type=str, default=None, help='batch_size to use')
  parser.add_argument('-f','--filter', dest='filter', type=float, nargs='+', default=0., help='batch_size to use')
  parser.add_argument('-d','--deltafilter', dest='deltafilter', type=int, default=-1, help='batch_size to use')
  parser.add_argument('-l','--lenfilter', dest='lenfilter', type=float, default=0., help='batch_size to use')
  parser.add_argument('--filter_score', type=str, default='METEOR',help='the input candidateJson')
  parser.add_argument('--filter_by', type=str, default='min',help='the input candidateJson')
  parser.add_argument('--show_samples', type=int, default=0,help='Show all samples of the primary model')
  parser.add_argument('--otherres', dest='otherres', type=str, nargs='+', default=[], help='generator/GAN checkpoint filename')
  parser.add_argument('--other_filter_by', type=str, default='max',help='the input candidateJson')
  parser.add_argument('--other_names', type=str, nargs='+',default=[],help='the input candidateJson')
  parser.add_argument('--model_name', type=str, default=None,help='the input candidateJson')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
