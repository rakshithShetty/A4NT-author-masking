import json
import os
import os.path as osp
from preproc_dataset import preproc_dataset
from collections import defaultdict
from collections import Counter
import torch
from torch import tensor
import numpy as np

class DataProvider():

    def __init__(self,params):
        dataset = params['dataset']
        datadir = osp.join('data',dataset)
        if osp.exists(osp.join(datadir,'dataset.json')):
            self.data = json.load(open(osp.join(datadir,'dataset.json'),'r'))
        else:
            self.data = preproc_dataset(osp.join(datadir,'splits','train'), datadir)

        self.splits = defaultdict(list)
        for i,dc in enumerate(self.data['docs']):
            self.splits[dc['split']].append(i)

        self.cur_char_idx = np.ones(len(self.data['docs']),dtype=np.int)
        self.cur_batch = np.array([-1]*params['batch_size'],dtype=np.int)
        self.max_seq_len = params['max_seq_len']

        return

    def iter_single_doc(self, split='train', max_docs=-1):
        # Since this is a standalone interation usually run to completion
        # we want to have a seperate temporary doc pointers here.
        cur_char_idx = np.ones(len(self.data['docs']),dtype=np.int)

        for i,cid in enumerate(self.splits[split]):
            if max_docs>= 0 and i >= max_docs:
                break
            dead = False
            while not dead:
                batch = []
                inp, targ, dead, cur_char_idx = self.get_nextstring_doc(cid, cur_char_idx)
                batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid]['author']})
                yield batch

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


    def get_doc_batch(self, split='train'):
        act_ids = np.where(self.cur_batch>=0)[0]
        dead_ids = np.where(self.cur_batch<0)[0]
        self.cur_batch[dead_ids] = np.random.choice(list(set(self.splits[split]) - set(act_ids.tolist())), len(dead_ids), replace=False)
        batch = []
        for i,cid in enumerate(self.cur_batch):
            inp, targ, dead, self.cur_char_idx = self.get_nextstring_doc(cid, self.cur_char_idx)
            self.cur_batch[i] = -1 if dead else cid
            batch.append({'in':inp,'targ': targ, 'author': self.data['docs'][cid]['author']})

        return batch, dead_ids

    def prepare_data(self, batch, char_to_ix, auth_to_ix):
        inp_seqs = []
        targ_seqs = []
        lens = []
        auths = []
        b_sz = len(batch)
        for b in batch:
            inp_seqs.append([char_to_ix[c] for c in b['in'] if c in char_to_ix])
            targ_seqs.append([char_to_ix[c] for c in b['targ'] if c in char_to_ix])
            lens.append(len(inp_seqs[-1]))
            if len(targ_seqs[-1]) != lens[-1]:
                if len(targ_seqs[-1]) < lens[-1]:
                    targ_seqs[-1].append(0)
                else:
                    inp_seqs[-1].append(0)
                    lens[-1] = lens[-1] + 1
            auths.append(auth_to_ix[b['author']])

        # pad the sequences
        max_len = max(lens)
        inp_seqs_arr = np.zeros((max_len, len(batch)), dtype=np.int)
        targ_seqs_arr = np.zeros((max_len, len(batch)), dtype=np.int)
        # Sort the sequences by length, highest first
        lens_arr = np.array(lens, dtype=np.int)
        sort_idx = np.argsort(lens_arr)[::-1]
        lens_arr = lens_arr[sort_idx]
        auths_arr = np.array(auths, dtype=np.int)[sort_idx]

        for i,j in enumerate(sort_idx):
            inp_seqs_arr[:lens_arr[i], i] = inp_seqs[j]
            targ_seqs_arr[:lens_arr[i], i] = targ_seqs[j]

        return torch.from_numpy(inp_seqs_arr), torch.from_numpy(targ_seqs_arr), torch.from_numpy(auths_arr), lens_arr.tolist()

    def createAuthorIdx(self):
        author_idx = {}
        n_authors = 0
        for i in self.splits['train']:
            if self.data['docs'][i]['author'] not in author_idx:
                author_idx[self.data['docs'][i]['author']] = n_authors
                n_authors = n_authors + 1

        return author_idx

    def createVocab(self, threshold=5):
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
        return minivocab, ixtochar