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
import torch.nn.functional as FN
import numpy as np

def sample_gumbel(x):
    noise = torch.rand(x.size()).cuda()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def gumbel_softmax_sample(x, tau=0.2, hard=False):
    noise = sample_gumbel(x)
    y = (x + noise) / tau
    y = FN.softmax(y)
    if hard:
        max_v, max_idx = y.max(dim=1,keepdim=True)
        one_hot = Variable(y.data.new(y.size()).zero_().scatter_(1, max_idx.data, y.data.new(max_idx.size()).fill_(1.)) - y.data, requires_grad=False)
        # Which is the right way to do this?
        #y_out = one_hot.detach() + y
        y_out = one_hot + y
        return y_out.view_as(x)
    return y.view_as(x)

class CharTranslator(nn.Module):
    def __init__(self, params):
        super(CharTranslator, self).__init__()
        #+1 is to allow padding index
        self.vocab_size = params.get('vocabulary_size',-1) + 1

        self.enc_num_rec_layers = params.get('enc_hidden_depth',-1)
        self.dec_num_rec_layers = params.get('dec_hidden_depth',-1)

        self.emb_size = params.get('embedding_size',-1)
        self.enc_hidden_size = params.get('enc_hidden_size',-1)
        self.dec_hidden_size = params.get('dec_hidden_size',-1)
        self.en_residual = params.get('en_residual_conn',1)

        # Initialize the model layers
        # Embedding layer
        self.char_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        self.emb_drop = nn.Dropout(p=params.get('drop_prob_emb',0.25))

        self.enc_drop = nn.Dropout(p=params.get('drop_prob_encoder',0.25))
        self.max_pool_rnn = params.get('maxpoolrnn',0)

        # Translator consists of an encoder - attention layer - decoder
        # Encoder is an lstm network with which takes character embedding vectors as input
        # Decoder is an lstm network which takes hidden states from encoder as input and
        # outputs series of characters

        # Encoder Lstm Layers
        if self.en_residual:
            self.enc_rec_layers = nn.ModuleList([nn.LSTM(self.emb_size, self.enc_hidden_size, 1) for i in xrange(self.enc_num_rec_layers)])
        else:
            self.enc_rec_layers = nn.LSTM(self.emb_size, self.enc_hidden_size, self.enc_num_rec_layers)

        self.pool_enc = params.get('maxpoolrnn',0)

        # Decoder Lstm Layers
        if self.en_residual:
            self.dec_rec_layers = nn.ModuleList([nn.LSTM((self.enc_hidden_size*(1+self.max_pool_rnn)+self.emb_size), self.dec_hidden_size, 1) for i in xrange(self.dec_num_rec_layers)])
        else:
            self.dec_rec_layers = nn.LSTM(self.enc_hidden_size*(1+self.max_pool_rnn)+ self.emb_size, self.dec_hidden_size, self.dec_num_rec_layers)

        # Output decoder layer
        self.decoder_W = nn.Parameter(torch.zeros([self.dec_hidden_size,
            self.vocab_size]), True)
        self.decoder_b = nn.Parameter(torch.zeros([self.vocab_size]), True)

        self.dec_drop = nn.Dropout(p=params.get('drop_prob_decoder',0.25))

        self.softmax = nn.LogSoftmax()

        self.init_weights()
        # we should move it out so that whether to do cuda or not should be upto the user.
        self.cuda()

    def init_weights(self):
        # Weight initializations for various parts.
        a = 0.001
        self.decoder_W.data.uniform_(-a, a)
        #self.encoder.weight.data.uniform_(-a, a)
        self.decoder_b.data.fill_(0)
        enc_h_sz = self.enc_hidden_size
        dec_h_sz = self.dec_hidden_size
        # LSTM forget gate could be initialized to high value (1.)
        if self.en_residual:
          for i in xrange(self.enc_num_rec_layers):
            self.enc_rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
            self.enc_rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
          for i in xrange(self.dec_num_rec_layers):
            self.dec_rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
            self.dec_rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
        else:
          for i in xrange(self.enc_num_rec_layers):
            getattr(self.enc_rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
            getattr(self.enc_rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)

          for i in xrange(self.dec_num_rec_layers):
            getattr(self.dec_rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
            getattr(self.dec_rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)

    def init_hidden(self, bsz):
        # Weight initializations for various parts.
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.enc_num_rec_layers, bsz, self.enc_hidden_size).zero_()),
                    Variable(weight.new(self.enc_num_rec_layers, bsz, self.enc_hidden_size).zero_()))

    def _my_recurrent_layer(self, packed, h_prev=None, rec_func=None, n_layers=1.):
        if self.en_residual:
            p_out = packed
            hid_all = []
            rec_func = self.rec_layers if rec_func == None else rec_func
            for i in xrange(n_layers):
                out, hid = rec_func[i](p_out, (h_prev[0][i:i+1], h_prev[1][i:i+1]))
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

    def forward_mltrain(self, inp, lengths_inp, targ, lengths_targ, h_prev, compute_softmax = False):
        # x should be a numpy array of n_seq x n_batch dimensions
        b_sz = inp.size(1)
        n_steps = inp.size(0)
        inp = Variable(inp).cuda()

        # Embed the sequence of characters
        emb = self.emb_drop(self.char_emb(inp))
        packed = pack_padded_sequence(emb, lengths_inp)

        # Encode the sequence of input characters
        enc_rnn_out, enc_hidden = self._my_recurrent_layer(packed, h_prev, rec_func = self.enc_rec_layers, n_layers = self.enc_num_rec_layers)

        enc_rnn_out_unp = pad_packed_sequence(enc_rnn_out)
        enc_rnn_out = self.enc_drop(enc_rnn_out_unp[0])

        if self.max_pool_rnn:
            ctxt = torch.cat([torch.mean(enc_rnn_out,dim=0,keepdim=False), enc_hidden[0][0]],
                dim=-1)
        else:
            ctxt = enc_hidden[0][0]

        # Setup target variable now
        targ = Variable(targ).cuda()
        targ_emb = self.emb_drop(self.char_emb(targ))
        # Concat the context vector from the encoder
        n_steps_targ = targ.size(0)

        dec_inp = torch.cat([ctxt.expand(n_steps_targ,b_sz,ctxt.size(1)), targ_emb], dim=-1)
        targ_packed = pack_padded_sequence(dec_inp, lengths_targ)

        # Decode the output sequence using encoder state
        dec_rnn_out, dec_hidden = self._my_recurrent_layer(targ_packed, h_prev=None, rec_func = self.dec_rec_layers,
                n_layers = self.dec_num_rec_layers)

        dec_out_unp = pad_packed_sequence(dec_rnn_out)
        dec_out = self.dec_drop(dec_out_unp[0])

        # implement the multi-headed RNN.
        W = self.decoder_W
        # reshape and expand b to size (batch*n_steps*vocab_size)
        b = self.decoder_b.view(1, self.vocab_size)
        b = b.expand(b_sz*n_steps_targ, self.vocab_size)
        # output is size seq * batch_size * vocab
        score_out = dec_out.view(n_steps_targ*b_sz,-1).mm(W) + b

        if compute_softmax:
            prob_out = self.softmax(score_out).view(n_steps_targ, b_sz, self.vocab_size)
        else:
            prob_out = score_out.view(n_steps_targ, b_sz, self.vocab_size)

        return prob_out, (enc_hidden, dec_hidden)
    
    def forward_advers_gen(self, x, lengths_inp, h_prev=None, n_max = 100, end_c = -1, soft_samples=False, temp=0.1):
        # Sample n_max characters give the hidden state and initial seed x.  Seed should have
        # atleast one character (eg. begin doc char), h_prev can be zeros. x is assumed to be
        # n_steps x 1 dimensional, i.e only one sample string generation at a time. Generation is
        # done using target author.

        n_steps = x.size(0)
        b_sz = x.size(1)
        if self.training:
            x = Variable(x).cuda()
        else:
            x = Variable(x,volatile=True).cuda()

        emb = self.emb_drop(self.char_emb(x))
        packed = pack_padded_sequence(emb, lengths_inp)

        # Encode the sequence of input characters
        enc_rnn_out, enc_hidden = self._my_recurrent_layer(packed, h_prev, rec_func = self.enc_rec_layers, n_layers = self.enc_num_rec_layers)
        
        if self.max_pool_rnn:
            ctxt = torch.cat([packed_mean(enc_rnn_out, dim=0), enc_hidden[0][0]],
                dim=-1)
        else:
            ctxt = enc_hidden[0][0]

        targ_init = x[0]
        targ_emb = self.char_emb(targ_init)

        dec_inp = torch.cat([ctxt, targ_emb], dim=-1)
        # Decode the output sequence using encoder state
        dec_rnn_out, dec_hidden = self._my_recurrent_layer(dec_inp.view(1,b_sz,-1), h_prev=None, rec_func = self.dec_rec_layers,
                n_layers = self.dec_num_rec_layers)

        # implement the multi-headed RNN.
        W = self.decoder_W
        # reshape and expand b to size (batch*n_steps*vocab_size)
        b = self.decoder_b.view(1, self.vocab_size).expand(b_sz, self.vocab_size)

        p_rnn = dec_rnn_out.view(b_sz,-1)
        char_out = []
        samp_out = []
        gen_lens = np.zeros(b_sz, dtype=np.int)
        prev_done = np.zeros(b_sz, dtype=np.bool)

        for i in xrange(n_max):
            # output is size seq * batch_size * vocab
            dec_out = p_rnn.mm(W) + b
            if  soft_samples:
                samp = gumbel_softmax_sample(dec_out*2.0, temp, hard=True)
                emb = samp.mm(self.char_emb.weight)
                _, pred_c = samp.data.max(dim=-1)
                char_out.append(pred_c)
                samp_out.append(samp)
            else:
                max_sc, pred_c = dec_out.data.max(dim=-1)
                char_out.append(pred_c)
                emb = self.char_emb(Variable(pred_c))

            gen_lens += (~prev_done)
            prev_done +=(pred_c.cpu().numpy() == end_c)
            if prev_done.all():
                break
            else:
                # No need for any packing here
                dec_inp = torch.cat([ctxt.view(1,b_sz, -1), emb.view(1,b_sz, -1)], dim=-1)
                p_rnn, dec_hidden = self._my_recurrent_layer(dec_inp, dec_hidden, rec_func = self.dec_rec_layers, n_layers = self.dec_num_rec_layers)
                p_rnn = p_rnn.view(b_sz, -1)

        return samp_out, gen_lens, char_out

    def forward_gen(self, x, h_prev=None, authors =None, n_max = 100, end_c = -1, soft_samples=False, temp=0.1):
        # Sample n_max characters give the hidden state and initial seed x.  Seed should have
        # atleast one character (eg. begin doc char), h_prev can be zeros. x is assumed to be
        # n_steps x 1 dimensional, i.e only one sample string generation at a time. Generation is
        # done using target author.

        n_steps = x.size(0)
        b_sz = x.size(1)
        if self.training:
            x = Variable(x).cuda()
        else:
            x = Variable(x,volatile=True).cuda()

        emb = self.emb_drop(self.char_emb(x))
        packed = emb

        # Encode the sequence of input characters
        enc_rnn_out, enc_hidden = self._my_recurrent_layer(packed, h_prev, rec_func = self.enc_rec_layers, n_layers = self.enc_num_rec_layers)

        if self.max_pool_rnn:
            ctxt = torch.cat([torch.mean(enc_rnn_out,dim=0,keepdim=False), enc_hidden[0][0]], dim=-1)
        else:
            ctxt = enc_hidden[0][0]

        targ_init = x[0]
        targ_emb = self.char_emb(targ_init)

        dec_inp = torch.cat([ctxt, targ_emb], dim=-1)

        # Decode the output sequence using encoder state
        dec_rnn_out, dec_hidden = self._my_recurrent_layer(dec_inp.view(1,1,-1), h_prev=None, rec_func = self.dec_rec_layers,
                n_layers = self.dec_num_rec_layers)

        # implement the multi-headed RNN.
        W = self.decoder_W
        # reshape and expand b to size (batch*n_steps*vocab_size)
        b = self.decoder_b.view(1, self.vocab_size)

        p_rnn = dec_rnn_out[-1]
        char_out = []

        for i in xrange(n_max):
            # output is size seq * batch_size * vocab
            dec_out = p_rnn.mm(W) + b
            if soft_samples:
                samp = gumbel_softmax_sample(dec_out, temp, hard=True)
                emb = samp.mm(self.char_emb.weight)
                char_out.append(samp)
            else:
                max_sc, pred_c = dec_out.max(dim=-1)
                char_out.append(pred_c)
                emb = self.char_emb(pred_c)

            if (pred_c == end_c).data[0]:
                break
            else:
                # No need for any packing here
                dec_inp = torch.cat([ctxt.view(1,1,-1), emb.view(1,1,-1)], dim=-1)
                p_rnn, dec_hidden = self._my_recurrent_layer(dec_inp, dec_hidden, rec_func = self.dec_rec_layers, n_layers = self.dec_num_rec_layers)
                p_rnn = p_rnn[-1]

        return char_out
    
