import json
import os
import os.path as osp
from preproc_dataset import preproc_dataset
from collections import defaultdict
from collections import Counter
import torch
from torch import tensor
import numpy as np
import random
from bisect import bisect

class DataProvider():

    def __init__(self,params):
        dataset = params['dataset']
        datadir = osp.join('data',dataset)
        if osp.exists(osp.join(datadir,params.get('dataset_file','dataset.json'))):
            self.data = json.load(open(osp.join(datadir,params.get('dataset_file','dataset.json')),'r'))
        else:
            self.data = preproc_dataset(osp.join(datadir,'splits','train'), datadir)

        self.athstr = 'author' if params.get('authstring',None) == None else params['authstring']

        if len(params.get('filterauths', [])) > 0:
            if params.get('filtertype','keep') == 'keep':
                keepset = set(params['filterauths'])
                print 'Keeping only %s'%keepset
                self.data['docs'] = [doc for doc in self.data['docs'] if str(doc[self.athstr]) in keepset]
            elif params.get('filtertype','keep') == 'agegroup':
                groupidces = map(int,params['filterauths'])
                for i,doc in enumerate(self.data['docs']):
                    grp = bisect(groupidces, doc['actage'])
                    self.data['docs'][i][self.athstr] = '<' + str(groupidces[grp]) if grp < len(groupidces) else 'None'
                self.data['docs'] = [doc for doc in self.data['docs'] if doc[self.athstr] != 'None']
            elif params.get('filtertype','keep') == 'agegroup-grt':
                groupidces = map(int,params['filterauths'])
                for i,doc in enumerate(self.data['docs']):
                    grp = bisect(groupidces, doc['actage'])
                    self.data['docs'][i][self.athstr] = '>' + str(groupidces[grp-1]) if grp > 0 else 'None'
                self.data['docs'] = [doc for doc in self.data['docs'] if doc[self.athstr] != 'None']

        self.splits = defaultdict(list)
        for i,dc in enumerate(self.data['docs']):
            self.splits[dc['split']].append(i)

        self.cur_batch = np.array([-1]*params.get('batch_size',1),dtype=np.int)
        self.max_seq_len = params.get('max_seq_len',100)

        # Some caching to track the current character index in the document and
        # hidden states associated with that. (Not sure if hidden states are still
        # relavant, they might need to be forgotten, esp. when network in changing
        # rapidly during initial stages of training.
        self.cur_char_idx = np.ones(len(self.data['docs']),dtype=np.int)

        self.hid_cache = {}

        self.use_unk = params.get('use_unk',0)

        if(dataset == 'blogdata'):
            self.min_len = 2
            self.max_len = 32
        elif(dataset == 'speechdata'):
            self.min_len = 2
            self.max_len = 30
        else:
            raise ValueError('ERROR: Dont know how to do len splitting for this dataset')

        self.lenMap = {}
        lenHist = {}
        for sp in self.splits:
            self.lenMap[sp] = defaultdict(list)
            lenHist[sp] = defaultdict(int)
            for iid in self.splits[sp]:
              doc = self.data['docs'][iid]
              for sid, tkn in enumerate(doc['tokens']):
                ix = max(min(len(tkn.split()),self.max_len),self.min_len)
                lenHist[sp][ix] += 1
                self.lenMap[sp][ix].append((iid,sid))
        self.min_len = min(lenHist['train'].keys())

        if not params.get('uniform_len_sample',0):
            self.lenCdist = np.cumsum(lenHist['train'].values())
        else:
            self.lenCdist = np.arange(len(lenHist['train']))+1
        return

    def set_hid_cache(self, idces, hid_state):
        # Limit the indexing to max items in hidden_states.
        #This allows setting multiple indices tosame vector
        n_hid = hid_state[0].data.size()[1]
        for i,cid in enumerate(idces):
            self.hid_cache[cid] = [torch.index_select(hd.data,1,torch.LongTensor([min(i,n_hid-1)]).cuda()) for hd in hid_state]
        return

    def get_hid_cache(self, idces, hid_state):
        for i in xrange(len(hid_state)):
            hid_state[i].data = torch.cat([self.hid_cache[cid][i] for cid in idces],dim=1)
        return hid_state

    def get_random_string(self, slen = 10, split='train', author=None):
        if author == None:
            author = random.choice(self.data['author-data'].keys())

        good_ids = list(set(self.splits[split]).intersection(set(self.data['author-data'][author]['idces'])))
        assert(len(good_ids)>0)

        idx = np.random.choice(good_ids,1)

        text = self.data['docs'][idx]['text']
        start_pos = np.random.randint(1,len(text)-slen)

        batch = []
        targ = text[start_pos:start_pos+slen]
        inp = text[start_pos-1:start_pos+slen-1]
        batch = [{'in':inp,'targ': targ, 'author': author}]
        return batch

    def iter_single_doc(self, split='train', max_docs=-1):
        # Since this is a standalone interation usually run to completion
        # we want to have a seperate temporary doc pointers here.
        cur_char_idx = np.ones(len(self.data['docs']),dtype=np.int)
        if max_docs > 0:
            idxes = np.random.choice(self.splits[split],max_docs)
        else:
            idxes = self.splits[split]


        for i,cid in enumerate(idxes):
            dead = False
            while not dead:
                batch = []
                inp, targ, dead, cur_char_idx = self.get_nextstring_doc(cid, cur_char_idx)
                batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid][self.athstr]})
                yield batch, dead

    def iter_sentences_bylen(self, split='train', batch_size=100, atoms='char', auths = None):
        # Since this is a standalone interation usually run to completion
        # we want to have a seperate temporary doc pointers here.
        sent_func = {'char':self.get_rand_sentence, 'word':self.get_rand_sentence_tokenized}
        batch = []
        for i,l in enumerate(self.lenMap[split]):
            for aid in auths:
                for idx in self.lenMap[split][l]:
                    if self.data['docs'][idx[0]][self.athstr] == aid:
                        inp, targ = sent_func[atoms](idx[0], sidx=idx[1])
                        batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][idx[0]][self.athstr], 'id':idx[0], 'sid':idx[1]})
                        if 'attrib' in self.data['docs'][idx[0]]:
                            batch[-1]['attrib'] = self.data['docs'][idx[0]]['attrib']
                    if len(batch) == batch_size:
                        yield batch, False
                        batch = []
                if batch:
                    yield batch, True
                    batch = []
            if batch:
                yield batch, True
                batch = []
        if batch:
            yield batch, True

    def iter_sentences(self, split='train', batch_size=100, atoms='char'):
        # Since this is a standalone interation usually run to completion
        # we want to have a seperate temporary doc pointers here.
        cur_char_idx = np.ones(len(self.data['docs']),dtype=np.int)
        sent_func = {'char':self.get_rand_sentence, 'word':self.get_rand_sentence_tokenized}
        batch = []

        for i,cid in enumerate(self.splits[split]):
            for j in xrange(self.get_num_sent_doc(cid, atoms=atoms)):
                inp, targ = sent_func[atoms](cid, sidx=j)
                batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid][self.athstr], 'id':cid, 'sid':j})
                if 'attrib' in self.data['docs'][cid]:
                    batch[-1]['attrib'] = self.data['docs'][cid]['attrib']
                if len(batch) == batch_size:
                    yield batch, j == (self.get_num_sent_doc(cid,atoms=atoms)-1)
                    batch = []
            if batch:
                yield batch, True
                batch = []

        if batch:
            yield batch, True

    def get_nextstring_doc(self, i, cur_char_idx, maxlen=-1):
        maxlen = self.max_seq_len if maxlen==-1 else maxlen
        cidx = cur_char_idx[i]
        txt = self.data['docs'][i]['text']

        eidx = cidx+maxlen if cidx+maxlen < len(txt) else len(txt)
        targ = txt[cidx:eidx]
        inp = txt[cidx-1:eidx-1]
        cur_char_idx[i] = eidx
        done = False

        if cur_char_idx[i] == len(txt):
            cur_char_idx[i] = 1
            done = True

        return inp, targ, done, cur_char_idx

    def get_num_seqs(self, maxlen=-1, split='train'):
        maxlen = self.max_seq_len if maxlen==-1 else maxlen
        total_batches = 0
        for i in self.splits[split]:
            total_batches = max(len(self.data['docs'][i]['text'])-1,0)//maxlen + 1 + total_batches
        return total_batches

    def get_num_sents(self, split='train'):
        total_batches = 0
        for i in self.splits[split]:
            total_batches = len(self.data['docs'][i]['tokens']) + total_batches
        return total_batches

    def get_num_sent_doc(self, cid, atoms = 'char'):
        sents = [st for st in self.data['docs'][cid]['text'].split('.') if len(st)>0] if atoms == 'char' else self.data['docs'][cid]['tokens']
        return len(sents)

    def get_doc_batch(self, split='train'):
        act_ids = np.where(self.cur_batch>=0)[0]
        dead_ids = np.where(self.cur_batch<0)[0]
        self.cur_batch[dead_ids] = np.random.choice(list(set(self.splits[split]) - set(act_ids.tolist())), len(dead_ids), replace=False)
        batch = []
        for i,cid in enumerate(self.cur_batch):
            inp, targ, dead, self.cur_char_idx = self.get_nextstring_doc(cid, self.cur_char_idx)
            self.cur_batch[i] = -1 if dead else cid
            batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid][self.athstr],'id':cid})

        return batch, dead_ids

    def get_rand_doc_batch(self, batch_size, split='train'):
        batch_ids = np.random.choice(self.splits[split], batch_size, replace=False)
        batch = []
        dead_ids_next_it = []
        for i,cid in enumerate(batch_ids):
            inp, targ, dead, self.cur_char_idx = self.get_nextstring_doc(cid, self.cur_char_idx)
            batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid][self.athstr],
                'id':cid})
            if dead:
                dead_ids_next_it.append(i)

        return batch, dead_ids_next_it

    def get_rand_sentence_tokenized(self, cid, sidx=None):
        sents = self.data['docs'][cid]['tokens']

        if len(sents) ==0:
            import ipdb;ipdb.set_trace()

        sidx = np.random.randint(0,len(sents),1) if sidx == None else sidx
        s = sents[sidx].split()

        targ = s[1:]
        inp = s[:-1]
        return inp, targ

    def get_rand_sentence(self, cid, sidx=None):
        sents = [st for st in self.data['docs'][cid]['text'].split('.') if len(st)>0]

        if len(sents) ==0:
            import ipdb;ipdb.set_trace()

        sidx = np.random.randint(0,len(sents),1) if sidx == None else sidx
        s = sents[sidx[0]]

        if s[0] == '2':
            targ = s[1:]+'.'
            inp = s
        else:
            targ = s + '.'
            inp = '2'+s
        return inp, targ

    def getRandLen(self, split='train'):
      """ sample image sentence pair from a split """

      rn = np.random.randint(0,self.lenCdist[-1])
      for l in xrange(len(self.lenCdist)):
          if rn < self.lenCdist[l] and (len(self.lenMap[split][l + self.min_len]) > 0):
              break

      l += self.min_len
      return l
  #def getRandBatchByLen(self,batch_size):
  #  """ sample image sentence pair from a split """

  #  rn = np.random.randint(0,self.lenCdist[-1])
  #  for l in xrange(len(self.lenCdist)):
  #      if rn < self.lenCdist[l] and (len(self.lenMap[l + self.min_len]) > 0):
  #          break

  #  l += self.min_len
  #  batch = [self.sampleImageSentencePairByLen(l) for i in xrange(batch_size)]
  #  return batch,l
    def get_sentence_batch(self, batch_size, split='train', atoms='char', aid=None, sample_by_len = False):
        allids = self.lenMap[split][self.getRandLen()] if sample_by_len else self.splits[split]
        if aid:
            allids = [idx for idx in allids if self.data['docs'][idx[0] if sample_by_len else idx][self.athstr] == aid]

        batch_ids = [allids[i] for i in np.random.randint(0, len(allids), batch_size)]
        batch = []
        sent_func = {'char':self.get_rand_sentence, 'word':self.get_rand_sentence_tokenized}
        for i,cids in enumerate(batch_ids):
            cid,sid = (cids) if sample_by_len else (cids,None)
            inp, targ = sent_func[atoms](cid,sid)
            batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid][self.athstr],
                'id':cid})
        return batch

    def prepare_data(self, batch, char_to_ix, auth_to_ix, leakage = 0., maxlen=None):
        inp_seqs = []
        targ_seqs = []
        lens = []
        auths = []
        b_sz = len(batch)
        for b in batch:
            inp_seqs.append([char_to_ix[c] for c in b['in'][:maxlen] if c in char_to_ix])
            targ_seqs.append([char_to_ix[c] for c in b['targ'][:maxlen] if c in char_to_ix])
            lens.append(len(inp_seqs[-1]))
            # Sometimes either the targ or inp might lose a character (OOV)
            # Handle that situation here.
            if len(targ_seqs[-1]) != lens[-1]:
                if len(targ_seqs[-1]) < lens[-1]:
                    targ_seqs[-1].append(0)
                else:
                    inp_seqs[-1].insert(0,0)
                    lens[-1] = lens[-1] + 1
            authidx = auth_to_ix[b['author']] if np.random.rand() >= leakage else np.random.choice(auth_to_ix.values())
            auths.append(authidx)

        # pad the sequences
        max_len = max(lens)
        inp_seqs_arr = np.zeros((max_len, len(batch)), dtype=np.int)
        targ_seqs_arr = np.zeros((max_len, len(batch)), dtype=np.int)
        # Sort the sequences by length, highest first
        lens_arr = np.array(lens, dtype=np.int)
        sort_idx = np.argsort(lens_arr)[::-1]
        lens_arr = lens_arr[sort_idx]
        auths_arr = np.array(auths, dtype=np.int)[sort_idx]
        # In place change the list so that batches are in the sorted order
        batch[:] = [batch[i] for i in sort_idx]


        for i,j in enumerate(sort_idx):
            inp_seqs_arr[:lens_arr[i], i] = inp_seqs[j]
            targ_seqs_arr[:lens_arr[i], i] = targ_seqs[j]

        return torch.from_numpy(inp_seqs_arr), torch.from_numpy(targ_seqs_arr), torch.from_numpy(auths_arr), lens_arr.tolist()

    def createAuthorIdx(self):
        author_idx = {}
        n_authors = 0
        for i in self.splits['train']:
            if self.data['docs'][i][self.athstr] not in author_idx:
                author_idx[self.data['docs'][i][self.athstr]] = n_authors
                n_authors = n_authors + 1

        ix_to_author = {author_idx[a]:a for a in author_idx}
        return author_idx, ix_to_author

    def createCharVocab(self, threshold=5):
        minivocab = {}
        ixtochar = {}
        vocab = Counter()
        for i in self.splits['train']:
            vocab.update([c for c in self.data['docs'][i]['text']])

        #+1  so that 0 is used for padding
        for i,c in enumerate(vocab):
            if vocab[c] >= threshold:
                minivocab[c] = len(minivocab) +1
                ixtochar[minivocab[c]] = c
        print 'Vocabulary size is %d'%(len(minivocab))
        return minivocab, ixtochar

    def createWordVocab(self, threshold=5):
        minivocab = {}
        ixtochar = {}
        vocab = Counter()
        for i in self.splits['train']:
            vocab.update([w for tk in self.data['docs'][i]['tokens'] for w in tk.split()])
        #+1  so that 0 is used for padding
        for i,c in enumerate(vocab):
            if vocab[c] >= threshold:
                minivocab[c] = len(minivocab) +1
                ixtochar[minivocab[c]] = c
        if self.use_unk:
            minivocab['UNK'] = len(minivocab) +1
            ixtochar[minivocab['UNK']]= 'UNK'
            print 'Replacing unknown tokens with UNK'
            for i,doc in enumerate(self.data['docs']):
                for j, tk in enumerate(doc['tokens']):
                    self.data['docs'][i]['tokens'][j] = ' '.join([w if w in minivocab else 'UNK' for w in tk.split()])

        print 'Vocabulary size is %d'%(len(minivocab))
        return minivocab, ixtochar
