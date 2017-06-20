import torch
import torchvision.models
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import tensor


class CharLstm(nn.Module):
    def __init__(self, params):
        super(CharLstm, self).__init__()
        #+1 is to allow padding index
        self.output_size = params.get('vocabulary_size',-1) + 1
        self.num_output_layers = params.get('num_output_layers',1)
        self.num_rec_layers = params.get('hidden_depth',-1)
        self.emb_size = params.get('embedding_size',-1)
        self.hidden_size = params.get('hidden_size',-1)
        self.en_residual = params.get('en_residual_conn',1)

        # Initialize the model layers
        # Embedding layer
        self.encoder = nn.Embedding(self.output_size, self.emb_size, padding_idx=0)
        self.enc_drop = nn.Dropout(p=params.get('drop_prob_encoder',0.25))

        # Lstm Layers
        if self.en_residual:
            self.rec_layers = nn.ModuleList([nn.LSTM(self.emb_size, self.hidden_size, 1) for i in xrange(self.num_rec_layers)])
        else:
            self.rec_layers = nn.LSTM(self.emb_size, self.hidden_size, self.num_rec_layers)

        self.max_pool_rnn = params.get('maxpoolrnn',0)

        # Output decoder layer
        if params.get('mode','generative')=='classifier':
            self.decoder_W = nn.Parameter(torch.zeros([self.hidden_size*(1+self.max_pool_rnn),
                self.num_output_layers]), True)
            self.decoder_b = nn.Parameter(torch.zeros([self.num_output_layers]), True)
        else:
            self.decoder_W = nn.Parameter(torch.zeros([self.num_output_layers, self.hidden_size,
                self.output_size]), True)
            self.decoder_b = nn.Parameter(torch.zeros([self.num_output_layers, self.output_size]), True)

        #self.decoder = nn.ModuleList([nn.Linear(self.hidden_size,self.output_size) for i in
        #                             xrange(self.num_output_layers)])
        self.dec_drop = nn.Dropout(p=params.get('drop_prob_decoder',0.25))

        if params['mode']=='generative' or 1:
            self.softmax = nn.LogSoftmax()
        else:
            self.softmax = nn.Softmax()

        self.init_weights()
        # we should move it out so that whether to do cuda or not should be upto the user.
        self.cuda()

    def init_weights(self):
        # Weight initializations for various parts.
        a = 0.01
        self.decoder_W.data.uniform_(-a, a)
        #self.encoder.weight.data.uniform_(-a, a)
        self.decoder_b.data.fill_(0)
        h_sz = self.hidden_size
        # LSTM forget gate could be initialized to high value (1.)
        if self.en_residual:
          for i in xrange(self.num_rec_layers):
            self.rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
            self.rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
        else:
          for i in xrange(self.num_rec_layers):
            getattr(self.rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 0.)
            getattr(self.rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 0.)

    def init_hidden(self, bsz):
        # Weight initializations for various parts.
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_rec_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_rec_layers, bsz, self.hidden_size).zero_()))

    def _my_recurrent_layer(self, packed, h_prev = None):
        if self.en_residual:
            p_out = packed
            hid_all = []
            for i in xrange(self.num_rec_layers):
                out, hid = self.rec_layers[i](p_out, (h_prev[0][i:i+1], h_prev[1][i:i+1]))
                if i > 0:
                    # Add residual connections after every layer
                    p_out = p_out + out
                else:
                    p_out = out
                hid_all.append(hid)
            hidden = (torch.cat([hid[0] for hid in hid_all],dim=-1), torch.cat([hid[1] for hid in hid_all],dim=-1))
            rnn_out = p_out
        else:
            if h_prev == None:
                rnn_out, hidden = rec_func(packed)
            else:
                rnn_out, hidden = rec_func(packed, h_prev)
        return rnn_out, hidden

    def forward(self, x, lengths, h_prev, target_head, compute_softmax = False):
        # x should be a numpy array of n_seq x n_batch dimensions
        b_sz = x.size(1)
        n_steps = x.size(0)
        x = Variable(x).cuda()
        emb = self.enc_drop(self.encoder(x))
        packed = pack_padded_sequence(emb, lengths)

        rnn_out, hidden = self._my_recurrent_layer(packed, h_prev)

        rnn_out_unp = pad_packed_sequence(rnn_out)
        rnn_out = self.dec_drop(rnn_out_unp[0])

        # implement the multi-headed RNN.
        W = self.decoder_W[target_head.cuda()]
        # reshape and expand b to size (batch*n_steps*vocab_size)
        b = self.decoder_b[target_head.cuda()].view(b_sz, -1, self.output_size)
        b = b.expand(b_sz, x.size(0), self.output_size)
        # output is size seq * batch_size * vocab
        dec_out = torch.baddbmm(b, rnn_out.transpose(0,1), W).transpose(0,1)

        if compute_softmax:
            prob_out = self.softmax(dec_out.view(-1, self.output_size)).view(n_steps, b_sz, self.output_size)
        else:
            prob_out = dec_out

        return prob_out, hidden

    def forward_eval(self, x, h_prev, compute_softmax = True):
        # x should be a numpy array of n_seq x n_batch dimensions
        # In this case batch will be a single sequence.
        n_auth = self.num_output_layers
        n_steps = x.size(0)
        x = Variable(x,volatile=True).cuda()
        # No Dropout needed
        emb = self.encoder(x)
        # No need for any packing here
        packed = emb

        rnn_out, hidden = self._my_recurrent_layer(packed, h_prev)

        # implement the multi-headed RNN.
        rnn_out = rnn_out.expand(n_steps, n_auth, self.hidden_size)
        W = self.decoder_W

        # reshape and expand b to size (n_auth*n_steps*vocab_size)
        b = self.decoder_b.view(n_auth, -1, self.output_size).expand(n_auth, n_steps, self.output_size)

        # output is size seq * batch_size * vocab
        dec_out = torch.baddbmm(b, rnn_out.transpose(0,1), W).transpose(0,1)

        if compute_softmax:
            prob_out = self.softmax(dec_out.contiguous().view(-1, self.output_size)).view(n_steps, n_auth, self.output_size)
        else:
            prob_out = dec_out

        return prob_out, hidden

    def forward_gen(self, x, h_prev, target_auth, n_max = 100, end_c = -1):
        # Sample n_max characters give the hidden state and initial seed x.  Seed should have
        # atleast one character (eg. begin doc char), h_prev can be zeros. x is assumed to be
        # n_steps x 1 dimensional, i.e only one sample string generation at a time. Generation is
        # done using target author.

        n_auth = self.num_output_layers
        n_steps = x.size(0)
        x = Variable(x,volatile=True).cuda()
        emb = self.encoder(x)
        # No need for any packing here
        packed = emb

        # Feed in the seed string. We are not intersted in these outputs except for the last one.
        rnn_out, hidden = self._my_recurrent_layer(packed, h_prev)

        W = self.decoder_W[target_auth.cuda()][0]
        # reshape and expand b to size (batch*n_steps*vocab_size)
        b = self.decoder_b[target_auth.cuda()].view(1, self.output_size)

        p_rnn = rnn_out[-1]
        char_out = []

        for i in xrange(n_max):
            # output is size seq * batch_size * vocab
            dec_out = p_rnn.mm(W) + b
            max_sc, pred_c = dec_out.max(dim=-1)
            char_out.append(pred_c)
            if 0:#pred_c == end_c:
                break
            else:
                emb = self.encoder(pred_c)
                # No need for any packing here
                packed = emb
                p_rnn, hidden = self._my_recurrent_layer(packed, hidden)
                p_rnn = p_rnn[-1]

        return char_out

    def forward_classify(self, x, h_prev=None, compute_softmax = False, predict_mode=False):
        # x should be a numpy array of n_seq x n_batch dimensions
        # In this case batch will be a single sequence.
        n_auth = self.num_output_layers
        n_steps = x.size(0)
        if predict_mode:
            x = Variable(x,volatile=True).cuda()
        else:
            x = Variable(x).cuda()

        b_sz = x.size(1)
        # No Dropout needed
        emb = self.enc_drop(self.encoder(x))
        # No need for any packing here
        packed = emb

        rnn_out, hidden = self._my_recurrent_layer(packed, h_prev)
        # implement the multi-headed RNN.
        if self.max_pool_rnn:
            rnn_out = self.dec_drop(torch.cat([torch.mean(rnn_out,dim=0, keepdim=False), rnn_out[-1]],
                dim=-1))
        else:
            rnn_out = self.dec_drop(rnn_out[-1])

        W = self.decoder_W
        # reshape and expand b to size (n_auth*n_steps*vocab_size)
        b = self.decoder_b.expand(b_sz, self.num_output_layers)

        dec_out = rnn_out.mm(W) + b

        if compute_softmax:
            prob_out = self.softmax(dec_out.contiguous().view(-1, self.num_output_layers))
        else:
            prob_out = dec_out

        return prob_out, hidden
