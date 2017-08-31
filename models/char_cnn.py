import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import tensor
import torch.nn.functional as FN
import numpy as np

class CharCNN(nn.Module):
    def __init__(self, params):
        super(CharCNN, self).__init__()
        #+1 is to allow padding index
        self.n_words = params.get('vocabulary_size',-1) + 1
        self.num_outputs = params.get('num_output_layers',1)
        self.emb_size = params.get('embedding_size',-1)

        # Initialize the model layers
        # Embedding layer
        self.encoder = nn.Embedding(self.n_words, self.emb_size, padding_idx=0)
        self.enc_drop = nn.Dropout(p=params.get('drop_prob_encoder',0.25))

        # Output decoder layer
        self.cnn_ks = params['decoder_cnn_ks']
        self.cnn_nfilt = params['decoder_cnn_nfilt']
        self.decoder_cnn = nn.ModuleList([nn.Conv1d(self.emb_size,self.cnn_nfilt, K,padding=K) for K in self.cnn_ks])
        self.decoder_cnnlayer = True
        decoder_size = self.cnn_nfilt*len(self.cnn_ks)

        self.decoder_W = nn.Parameter(torch.zeros([decoder_size, self.num_outputs]), True)
        self.decoder_b = nn.Parameter(torch.zeros([self.num_outputs]), True)

        #self.decoder = nn.ModuleList([nn.Linear(self.hidden_size,self.output_size) for i in
        #                             xrange(self.num_outputs)])
        self.dec_drop = nn.Dropout(p=params.get('drop_prob_decoder',0.25))

        self.softmax = nn.Softmax()

        self.init_weights()
        # we should move it out so that whether to do cuda or not should be upto the user.
        self.cuda()

    def init_weights(self):
        # Weight initializations for various parts.
        a = np.sqrt(float(self.decoder_W.size(0)))
        self.decoder_W.data.uniform_(-1.73/a, 1.73/a)
        #self.encoder.weight.data.uniform_(-a, a)
        self.decoder_b.data.fill_(0)

    def init_hidden(self, bsz):
        # Weight initializations for various parts.
        return None

    def forward_classify(self, x, h_prev=None, compute_softmax = False, predict_mode=False, adv_inp=False, lens=None):
        # x should be a numpy array of n_seq x n_batch dimensions
        # In this case batch will be a single sequence.
        n_auth = self.num_outputs
        n_steps = x.size(0)
        b_sz = x.size(1)
        if not adv_inp:
            if predict_mode:
                x = Variable(x,volatile=True).cuda()
            else:
                x = Variable(x).cuda()

            emb = self.enc_drop(self.encoder(x))
        else:
            emb = self.enc_drop(x.view(n_steps*b_sz,-1).mm(self.encoder.weight).view(n_steps,b_sz, -1))

        W = self.decoder_W
        # reshape and expand b to size (n_auth*n_steps*vocab_size)
        b = self.decoder_b.expand(b_sz, self.num_outputs)

        emb_sorted = emb.permute(1,2,0)
        cnn_out = [FN.leaky_relu(conv(emb_sorted)) for conv in self.decoder_cnn]
        cnn_pool_out = [FN.max_pool1d(cn, cn.size(2)).squeeze(2) for cn in cnn_out]
        dec_in = self.dec_drop(torch.cat(cnn_pool_out, dim=1))
        enc_out = dec_in

        dec_out = dec_in.mm(W) + b
        if compute_softmax:
            prob_out = self.softmax(dec_out.contiguous().view(-1, self.num_outputs))
        else:
            prob_out = dec_out

        return prob_out, enc_out
