import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import tensor
import torch.nn.functional as FN
import numpy as np
from model_utils import packed_mean, packed_add

def sample_gumbel(x):
    noise = torch.cuda.FloatTensor(x.size()).uniform_()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def gumbel_softmax_sample(x, tau=0.2, hard=False):
    noise = sample_gumbel(x)
    y = (FN.log_softmax(x) + noise) / tau
    ysft = FN.softmax(y)
    if hard:
        max_v, max_idx = ysft.max(dim=1,keepdim=True)
        one_hot = Variable(ysft.data.new(ysft.size()).zero_().scatter_(1, max_idx.data, ysft.data.new(max_idx.size()).fill_(1.)) - ysft.data, requires_grad=False)
        # Which is the right way to do this?
        #y_out = one_hot.detach() + y
        y_out = one_hot + ysft
        return y_out.view_as(x)
    return ysft.view_as(x)

class CharTranslator(nn.Module):
    def __init__(self, params, encoder_only=False):
        super(CharTranslator, self).__init__()
        #+1 is to allow padding index
        self.encoder_only = encoder_only
        self.encoder_mean_vec = params.get('encoder_mean_vec', 0)
        self.vocab_size = params.get('vocabulary_size',-1) + 1
        self.no_encoder = params.get('no_encoder',0)

        self.enc_num_rec_layers = params.get('enc_hidden_depth',-1)
        self.enc_noise = params.get('apply_noise',0)

        self.emb_size = params.get('embedding_size',-1)
        self.enc_hidden_size = params.get('enc_hidden_size',-1)
        self.dec_num_rec_layers = params.get('dec_hidden_depth',-1)
        self.dec_hidden_size = params.get('dec_hidden_size',-1)
        self.en_residual = params.get('en_residual_conn',1)
        self.pad_auth_vec = params.get('pad_auth_vec',0)

        self.gumb_type = params.get('gumbel_hard', False)
        self.learn_gumbel = params.get('learn_gumbel', False)
        self.softmax_scale = params.get('softmax_scale', 2.0)
        self.split_gen = params.get('split_generators', 0)

        if self.learn_gumbel:
           self.gumbel_W = nn.Parameter(torch.zeros([self.dec_hidden_size,
               1]), True)
           self.gumbel_b = nn.Parameter(torch.zeros([1]), True)


        # Initialize the model layers
        if self.pad_auth_vec:
            self.auth_emb = nn.Embedding(params.get('num_output_layers',2), self.pad_auth_vec)
        # Embedding layer
        self.char_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        self.emb_drop = nn.Dropout(p=params.get('drop_prob_emb',0.25))

        # Translator consists of an encoder - attention layer - decoder
        # Encoder is an lstm network with which takes character embedding vectors as input
        # Decoder is an lstm network which takes hidden states from encoder as input and
        # outputs series of characters

        # Encoder Lstm Layers
        if not self.no_encoder:
            if self.en_residual:
                inp_sizes = [self.emb_size] + [self.enc_hidden_size]*(self.enc_num_rec_layers-1)
                self.enc_rec_layers = nn.ModuleList([nn.LSTM(inp_sizes[i], self.enc_hidden_size, 1) for i in xrange(self.enc_num_rec_layers)])
            else:
                self.enc_rec_layers = nn.LSTM(self.emb_size, self.enc_hidden_size, self.enc_num_rec_layers)
            self.enc_drop = nn.Dropout(p=params.get('drop_prob_encoder',0.25))
            self.max_pool_rnn = params.get('maxpoolrnn',0)


        if not encoder_only:
            # Decoder Lstm Layers
            enc_out_size = self.enc_hidden_size*(1+self.max_pool_rnn) if not self.no_encoder else 0
            input_layer_size = (enc_out_size+self.emb_size + self.pad_auth_vec)
            if self.en_residual:
                self.dec_rec_layers = nn.ModuleList([nn.LSTM(input_layer_size if i == 0 else self.dec_hidden_size, self.dec_hidden_size, 1) for i in xrange(self.dec_num_rec_layers)])
            else:
                self.dec_rec_layers = nn.LSTM(input_layer_size, self.dec_hidden_size, self.dec_num_rec_layers)

            # Output decoder layer
            self.decoder_W = nn.Parameter(torch.zeros([self.dec_hidden_size,
                self.vocab_size]), True)
            self.decoder_b = nn.Parameter(torch.zeros([self.vocab_size]), True)
            self.dec_drop = nn.Dropout(p=params.get('drop_prob_decoder',0.25))

            # Assumes only for 2 authors. And with this a batch can only contain one author type
            if self.split_gen:
                if self.en_residual:
                    self.dec_rec_layers_2 = nn.ModuleList([nn.LSTM(input_layer_size if i == 0 else self.dec_hidden_size, self.dec_hidden_size, 1) for i in xrange(self.dec_num_rec_layers)])
                else:
                    self.dec_rec_layers_2 = nn.LSTM(input_layer_size, self.dec_hidden_size, self.dec_num_rec_layers)

                # Output decoder layer
                self.decoder_W_2 = nn.Parameter(torch.zeros([self.dec_hidden_size,
                    self.vocab_size]), True)
                self.decoder_b_2 = nn.Parameter(torch.zeros([self.vocab_size]), True)

            #if params_

            self.softmax = nn.LogSoftmax()

        self.init_weights()
        # we should move it out so that whether to do cuda or not should be upto the user.
        self.cuda()
        self.zero_hidden_bsz = params['batch_size'] # Create a dummy zero hidden state so it can be passed to lstm
        self.zero_hidden = self.init_hidden(params['batch_size']) # Create a dummy zero hidden state so it can be passed to lstm
        self.zero_hidden_dec = self.init_hidden_dec(params['batch_size']) # Create a dummy zero hidden state so it can be passed to lstm

    def init_weights(self):
        # Weight initializations for various parts.
        if not self.encoder_only:
            a = 0.001
            self.decoder_W.data.uniform_(-a, a)
            #self.encoder.weight.data.uniform_(-a, a)
            self.decoder_b.data.fill_(0)
            if self.split_gen:
                self.decoder_W_2.data.uniform_(-a, a)
                self.decoder_b_2.data.fill_(0)

        if self.learn_gumbel:
           self.gumbel_W.data.uniform_(-0.01, 0.01)

        enc_h_sz = self.enc_hidden_size
        dec_h_sz = self.dec_hidden_size
        # LSTM forget gate could be initialized to high value (1.)
        if self.en_residual:
          if not self.no_encoder:
            for i in xrange(self.enc_num_rec_layers):
              self.enc_rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
              self.enc_rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
          if not self.encoder_only:
            for i in xrange(self.dec_num_rec_layers):
              self.dec_rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
              self.dec_rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
              if self.split_gen:
                  self.dec_rec_layers_2[i].bias_ih_l0.data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
                  self.dec_rec_layers_2[i].bias_hh_l0.data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
        else:
          if not self.no_encoder:
            for i in xrange(self.enc_num_rec_layers):
              getattr(self.enc_rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
              getattr(self.enc_rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(enc_h_sz +1, enc_h_sz*2).long(), 1.)
          if not self.encoder_only:
            for i in xrange(self.dec_num_rec_layers):
              getattr(self.dec_rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
              getattr(self.dec_rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
              if self.split_gen:
                  getattr(self.dec_rec_layers_2,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)
                  getattr(self.dec_rec_layers_2,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(dec_h_sz +1, dec_h_sz*2).long(), 1.)

    def init_hidden(self, bsz):
        # Weight initializations for various parts.
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.enc_num_rec_layers, bsz, self.enc_hidden_size).zero_()),
                    Variable(weight.new(self.enc_num_rec_layers, bsz, self.enc_hidden_size).zero_()))

    def init_hidden_dec(self, bsz):
        # Weight initializations for various parts.
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.dec_num_rec_layers, bsz, self.dec_hidden_size).zero_()).detach(),
                    Variable(weight.new(self.dec_num_rec_layers, bsz, self.dec_hidden_size).zero_()).detach())

    def _my_recurrent_layer(self, packed, h_prev=None, rec_func=None, n_layers=1.):
        if self.en_residual:
            p_out = packed
            hid_all = []
            rec_func = self.rec_layers if rec_func == None else rec_func
            for i in xrange(n_layers):
                if h_prev == None :
                    out, hid = rec_func[i](p_out)
                else:
                    out, hid = rec_func[i](p_out, (h_prev[0][i:i+1], h_prev[1][i:i+1]))
                if i > 0:
                    # Add residual connections after every layer
                    p_out = packed_add(p_out, out) if type(p_out) == torch.nn.utils.rnn.PackedSequence else p_out + out
                else:
                    p_out = out
                hid_all.append(hid)
            hidden = (torch.cat([hid[0] for hid in hid_all],dim=0), torch.cat([hid[1] for hid in hid_all],dim=0))
            rnn_out = p_out
        else:
            if h_prev == None :
                rnn_out, hidden = rec_func(packed)
            else:
                rnn_out, hidden = rec_func(packed, h_prev)
        return rnn_out, hidden

    def forward_mltrain(self, inp, lengths_inp, targ, lengths_targ, h_prev=None, compute_softmax = False, auths=None, adv_inp=False, sort_enc=None, adv_targ=False):
        # x should be a numpy array of n_seq x n_batch dimensions
        b_sz = inp.size(1)
        n_steps = inp.size(0)

        if not self.no_encoder:
            if not adv_inp:
                if self.training:
                    inp = Variable(inp).cuda()
                else:
                    inp = Variable(inp,volatile=True).cuda()

                emb = self.emb_drop(self.char_emb(inp))
            else:
                emb = inp.view(n_steps*b_sz,-1).mm(self.char_emb.weight).view(n_steps,b_sz, -1)

            # Embed the sequence of characters
            packed = pack_padded_sequence(emb, lengths_inp)

            # Encode the sequence of input characters
            h_prev_enc = h_prev if h_prev !=None or b_sz != self.zero_hidden_bsz else self.zero_hidden
            enc_rnn_out, enc_hidden = self._my_recurrent_layer(packed, h_prev=h_prev_enc, rec_func = self.enc_rec_layers, n_layers = self.enc_num_rec_layers)

            enc_rnn_out_unp = pad_packed_sequence(enc_rnn_out)
            enc_rnn_out = enc_rnn_out_unp[0]

            if self.max_pool_rnn:
                ctxt = torch.cat([torch.mean(enc_rnn_out,dim=0,keepdim=False), enc_hidden[0][-1]],
                    dim=-1)
            else:
                ctxt = enc_hidden[0][-1]

            if type(sort_enc) != type(None):
                ctxt_sorted = ctxt.index_select(0,sort_enc)
            else:
                ctxt_sorted = ctxt

            if self.pad_auth_vec:
                ctxt_sorted = torch.cat([ctxt_sorted,self.auth_emb(Variable(auths).cuda())], dim=-1)

            if self.enc_noise:
                ctxt_sorted = ctxt_sorted + Variable(torch.cuda.FloatTensor(ctxt_sorted.size()).normal_()/20., requires_grad=False)
            else:
                ctxt_sorted = self.enc_drop(ctxt_sorted)
        else:
            enc_hidden = None

        # Setup target variable now
        if not adv_targ:
            targ = Variable(targ).cuda()
            targ_emb = self.emb_drop(self.char_emb(targ))
            n_steps_targ = targ.size(0)
        else:
            targ_emb = self.emb_drop(targ.view(n_steps*b_sz,-1).mm(self.char_emb.weight).view(n_steps,b_sz, -1))
            n_steps_targ = lengths_targ[0]
        # Concat the context vector from the encoder

        if not self.no_encoder:
            dec_inp = torch.cat([ctxt_sorted.expand(n_steps_targ,b_sz,ctxt_sorted.size(1)), targ_emb], dim=-1)
        else:
            dec_inp = targ_emb
        targ_packed = pack_padded_sequence(dec_inp, lengths_targ)

        # Decode the output sequence using encoder state
        h_prev_dec = None if b_sz != self.zero_hidden_bsz else self.zero_hidden_dec
        dec_rec_func = self.dec_rec_layers if not self.split_gen or auths[0] == 0 else self.dec_rec_layers_2
        dec_rnn_out, dec_hidden = self._my_recurrent_layer(targ_packed, h_prev=h_prev_dec, rec_func = dec_rec_func,
                n_layers = self.dec_num_rec_layers)

        dec_out_unp = pad_packed_sequence(dec_rnn_out)
        dec_out = self.dec_drop(dec_out_unp[0])

        # implement the multi-headed RNN.
        W = self.decoder_W if not self.split_gen or auths[0] == 0 else self.decoder_W_2
        # reshape and expand b to size (batch*n_steps*vocab_size)
        decoder_b = self.decoder_b if not self.split_gen or auths[0] == 0 else self.decoder_b_2
        b = decoder_b.view(1, self.vocab_size)
        b = b.expand(b_sz*n_steps_targ, self.vocab_size)
        # output is size seq * batch_size * vocab
        score_out = dec_out.view(n_steps_targ*b_sz,-1).mm(W) + b

        if compute_softmax:
            prob_out = self.softmax(score_out).view(n_steps_targ, b_sz, self.vocab_size)
        else:
            prob_out = score_out.view(n_steps_targ, b_sz, self.vocab_size)

        return prob_out, (enc_hidden, dec_hidden)

    def forward_encode(self, x, lengths_inp, h_prev=None, adv_inp=False):
        n_steps = x.size(0)
        b_sz = x.size(1)
        if not adv_inp:
            if self.training:
                x = Variable(x).cuda()
            else:
                x = Variable(x,volatile=True).cuda()

            emb = self.char_emb(x)
        else:
            emb = x.view(n_steps*b_sz,-1).mm(self.char_emb.weight).view(n_steps,b_sz, -1)
        packed = pack_padded_sequence(emb, lengths_inp)


        h_prev_enc = h_prev if h_prev !=None or b_sz != self.zero_hidden_bsz else self.zero_hidden
        # Encode the sequence of input characters
        enc_rnn_out, enc_hidden = self._my_recurrent_layer(packed, h_prev=h_prev_enc, rec_func = self.enc_rec_layers, n_layers = self.enc_num_rec_layers)

        if self.encoder_mean_vec:
            ctxt = torch.cat([packed_mean(enc_rnn_out, dim=0), enc_hidden[0][-1]],
                dim=-1)
        else:
            ctxt = enc_hidden[0][-1]
        return ctxt

    def forward_advers_gen(self, x, lengths_inp, h_prev=None, n_max = 100, end_c = -1, soft_samples=False, temp=0.1, auths=None, adv_inp=False, n_samples=1):
        # Sample n_max characters give the hidden state and initial seed x.  Seed should have
        # atleast one character (eg. begin doc char), h_prev can be zeros. x is assumed to be
        # n_steps x 1 dimensional, i.e only one sample string generation at a time. Generation is
        # done using target author.
        n_steps = x.size(0)
        b_sz = x.size(1)
        if not adv_inp:
            if self.training:
                x = Variable(x).cuda()
            else:
                x = Variable(x,volatile=True).cuda()

            emb = self.char_emb(x)
        else:
            emb = x.view(n_steps*b_sz,-1).mm(self.char_emb.weight).view(n_steps,b_sz, -1)
        packed = pack_padded_sequence(emb, lengths_inp)


        h_prev_enc = h_prev if h_prev !=None or b_sz != self.zero_hidden_bsz else self.zero_hidden
        # Encode the sequence of input characters
        enc_rnn_out, enc_hidden = self._my_recurrent_layer(packed, h_prev=h_prev_enc, rec_func = self.enc_rec_layers, n_layers = self.enc_num_rec_layers)

        if self.max_pool_rnn:
            ctxt = torch.cat([packed_mean(enc_rnn_out, dim=0), enc_hidden[0][-1]],
                dim=-1)
        else:
            ctxt = enc_hidden[0][-1]

        #Append with target author embedding
        if self.pad_auth_vec:
            ctxt = torch.cat([ctxt,self.auth_emb(Variable(auths).cuda())], dim=-1)

        if not adv_inp:
            targ_init = x[0]
            targ_emb = self.char_emb(targ_init)
        else:
            targ_emb = x[0,:,:].mm(self.char_emb.weight)

        dec_inp = torch.cat([ctxt, targ_emb], dim=-1).repeat(n_samples,1)

        dec_bsz = b_sz * n_samples

        h_prev_dec = None if dec_bsz != self.zero_hidden_bsz else self.zero_hidden_dec
        # Decode the output sequence using encoder state
        dec_rec_func = self.dec_rec_layers if not self.split_gen or auths[0] == 0 else self.dec_rec_layers_2
        dec_rnn_out, dec_hidden = self._my_recurrent_layer(dec_inp.view(1,dec_bsz,-1), h_prev=h_prev_dec, rec_func = dec_rec_func,
                n_layers = self.dec_num_rec_layers)

        # implement the multi-headed RNN.
        W = self.decoder_W if not self.split_gen or auths[0] == 0 else self.decoder_W_2
        # reshape and expand b to size (batch*n_steps*vocab_size)
        decoder_b = self.decoder_b if not self.split_gen or auths[0] == 0 else self.decoder_b_2
        b = decoder_b.view(1, self.vocab_size).expand(dec_bsz, self.vocab_size)

        p_rnn = dec_rnn_out.view(dec_bsz,-1)
        char_out = []
        samp_out = []
        gen_lens = torch.cuda.IntTensor(dec_bsz).zero_()
        prev_done = torch.cuda.ByteTensor(dec_bsz).zero_()

        for i in xrange(n_max):
            # output is size seq * batch_size * vocab
            dec_out = p_rnn.mm(W) + b
            if soft_samples:
                self.temp = (FN.softplus(p_rnn.mm(self.gumbel_W)+self.gumbel_b) + 1.) if self.learn_gumbel else temp
                samp = gumbel_softmax_sample(dec_out*self.softmax_scale, self.temp, hard=self.gumb_type)
                emb = samp.mm(self.char_emb.weight)
                _, pred_c = samp.data.max(dim=-1)
                char_out.append(pred_c)
                samp.data.masked_fill_(prev_done.view(-1,1), 0.)
                samp_out.append(samp)
            else:
                max_sc, pred_c = dec_out.data.max(dim=-1)
                char_out.append(pred_c)
                emb = self.char_emb(Variable(pred_c))

            gen_lens += (prev_done==0).int()
            prev_done +=(pred_c == end_c)
            if prev_done.all():
                break
            else:
                # No need for any packing here
                dec_inp = torch.cat([ctxt.repeat(n_samples,1).view(1,dec_bsz, -1), emb.view(1,dec_bsz, -1)], dim=-1)
                p_rnn, dec_hidden = self._my_recurrent_layer(dec_inp, h_prev=dec_hidden, rec_func = dec_rec_func, n_layers = self.dec_num_rec_layers)
                p_rnn = p_rnn.view(dec_bsz, -1)

        return samp_out, gen_lens, char_out

    def forward_gen(self, x, h_prev=None, auths=None, n_max = 100, end_c = -1, soft_samples=False, temp=0.1):
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
            ctxt = torch.cat([torch.mean(enc_rnn_out,dim=0,keepdim=False), enc_hidden[0][-1]], dim=-1)
        else:
            ctxt = enc_hidden[0][-1]
        #Append with target author embedding
        if self.pad_auth_vec:
            ctxt = torch.cat([ctxt,self.auth_emb(Variable(auths).cuda())], dim=-1)

        targ_init = x[0]
        targ_emb = self.char_emb(targ_init)

        dec_inp = torch.cat([ctxt, targ_emb], dim=-1)

        # Decode the output sequence using encoder state
        dec_rec_func = self.dec_rec_layers if not self.split_gen or auths[0] == 0 else self.dec_rec_layers_2
        dec_rnn_out, dec_hidden = self._my_recurrent_layer(dec_inp.view(1,1,-1), h_prev=None, rec_func = dec_rec_func,
                n_layers = self.dec_num_rec_layers)

        # implement the multi-headed RNN.
        W = self.decoder_W if not self.split_gen or auths[0] == 0 else self.decoder_W_2
        # reshape and expand b to size (batch*n_steps*vocab_size)
        decoder_b = self.decoder_b if not self.split_gen or auths[0] == 0 else self.decoder_b_2
        b = decoder_b.view(1, self.vocab_size)

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
                p_rnn, dec_hidden = self._my_recurrent_layer(dec_inp, dec_hidden, rec_func = dec_rec_func, n_layers = self.dec_num_rec_layers)
                p_rnn = p_rnn[-1]

        return char_out

