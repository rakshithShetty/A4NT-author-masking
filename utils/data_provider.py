import json
import os
import os.path as osp
from preproc_dataset import preproc_dataset
from collections import defaultdict
from collections import Counter

class DataProvider():

    def __init__(self,params):
        dataset = params['dataset']
        datadir = osp.join('data',dataset)
        if osp.exists(osp.join(datadir,'dataset.json')):
            self.data = json.load(open(osp.join(datadir,'dataset.json'),'r'))
        else:
            self.data = preproc_dataset(osp.join(datadir,'splits','train'), datadir)

        self.splits = defaultdict(splits)
        for i,dc in enumerate(data['docs']):
            self.splits[dc['split']].append(i)

        self.curr_char_idx = np.ones(len(data['docs']),dtype=np.int)
        self.curr_batch = np.array([-1]*params['batch_size'],dtype=np.int)
        self.max_seq_len = params['max_seq_len']

        return

    def get_nextstring_doc(self, i, maxlen=self.max_seq_len):
        cidx = self.cur_char_idx[i]
        txt = self.data['docs'][i]['text']

        eidx = cidx+maxlen if cidx+maxlen < len(txt) else len(txt)
        targ = txt[cidx:eidx]
        inp = txt[cidx-1:eidx-1]
        self.cur_char_idx[i] = eidx
        done = False

        if self.cur_char_idx[i] == len(txt):
            self.cur_char_idx[i] = 1
            done = True

        return inp, targ, done

    def get_doc_batch(self, split='train'):
        act_ids = np.where(self.cur_batch>=0)[0]
        dead_ids = np.where(self.cur_batch<0)[0]
        self.cur_batch[dead_ids] = np.random.choice(list(set(self.splits[split]) - set(act_ids.tolist())), len(dead_ids), replace=False)
        batch = []
        for i,cid in enumerate(self.cur_batch):
            inp, targ, dead = get_nextstring_doc(cid)
            self.cur_batch[i] = -1 if dead else cid
            batch.append({'in':inp,'targ': targ, 'auth': self.data['docs'][cid]['author']})

        return batch, dead_ids

    def prepare_data(self, batch, char_to_ix, auth_to_ix):
        inp_seqs = []
        targ_seqs = []
        lens = []
        auths = []
        for b in batch:
            inp_seqs.append([char_to_ix[c] for c in b['in'])
            targ_seqs.append([char_to_ix[c] for c in b['targ'])
            lens.append(len(inp_seqs[-1]))
            auths.append(auth_to_ix[b['author']])

        inp_seqs = np.array(inp_seqs)
        targ_seqs = np.array(targ_seqs)
        lens = np.array(lens)
        auths = np.array(auths)

        return inp_seqs, targ_seqs, auths, lens

    def createVocab(self, threshold=5):
        vocab = Counter()
        for i in self.splits['train']:
            vocab.update([c for c in self.docs[i]['text']])

        vocab
