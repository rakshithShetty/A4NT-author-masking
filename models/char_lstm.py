import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import tensor
from model_utils import packed_mean, packed_add
import torch.nn.functional as FN
import numpy as np
from torch.autograd import Variable, Function


class GradEmbMod(Function):
    def __init__(self):
        super(GradEmbMod, self).__init__()

    def forward(self, x, weight):
        b_sz = x.size(1)
        n_steps = x.size(0)
        emb_mul = (x.view(n_steps*b_sz,-1).mm(weight).view(n_steps,b_sz, -1))
        self.save_for_backward(x,weight, emb_mul)
        return emb_mul

    def backward(self, grad_output):
        import ipdb;ipdb.set_trace()
        n_time, b_sz = grad_output.size()[:2]
        _, topkidx = grad_output.abs().sum(dim=-1).topk(self.topk,dim=0)
        mask = torch.zeros(n_time, b_sz).cuda().scatter_(0,topkidx, 1.)
        grad_output.mul_(mask.unsqueeze(-1))
        return grad_output


class CharLstm(nn.Module):
    def __init__(self, params):
        super(CharLstm, self).__init__()
        #+1 is to allow padding index
        self.output_size = params.get('vocabulary_size',-1) + 1
        self.num_output_layers = params.get('num_output_layers',1)
        self.num_rec_layers = params.get('hidden_depth',-1)
        self.emb_size = params.get('embedding_size',-1)
        self.hidden_size = params.get('hidden_size',-1)
        self.en_residual = params.get('en_residual_conn',0)
        self.bidir = params.get('bidir',0)
        self.compression_layer = params.get('compression_layer',0)

        # Initialize the model layers
        # Embedding layer
        if self.compression_layer:
            self.compression_W = nn.Embedding(self.output_size, self.compression_layer)
            self.encoder = nn.Embedding(self.compression_layer, self.emb_size)
        else:
            self.encoder = nn.Embedding(self.output_size, self.emb_size, padding_idx=0)
        self.enc_drop = nn.Dropout(p=params.get('drop_prob_encoder',0.25))

        # Lstm Layers
        if self.en_residual:
            inp_sizes = [self.emb_size] + [self.hidden_size]*(self.num_rec_layers-1)
            self.rec_layers = nn.ModuleList([nn.LSTM(inp_sizes[i], self.hidden_size, 1, bidirectional = self.bidir) for i in xrange(self.num_rec_layers)])
        else:
            self.rec_layers = nn.LSTM(self.emb_size, self.hidden_size, self.num_rec_layers, bidirectional = self.bidir)

        self.max_pool_rnn = params.get('maxpoolrnn',0)

        # Output decoder layer
        if params.get('mode','generative')=='classifier':
            decoder_size = self.hidden_size*(1+((self.max_pool_rnn==1)|(self.max_pool_rnn==3))) * (1+self.bidir)
            if params.get('decoder_mlp', 0):
                self.decoder_W_mlp = nn.Parameter(torch.zeros([decoder_size,
                    params['decoder_mlp']]), True)
                self.decoder_b_mlp = nn.Parameter(torch.zeros([params['decoder_mlp']]), True)
                decoder_size = params['decoder_mlp']
                self.decoder_mlp = True
            elif params.get('decoder_cnn',0):
                self.cnn_ks = params['decoder_cnn_ks']
                self.cnn_nfilt = params['decoder_cnn_nfilt']
                self.decoder_cnn = nn.ModuleList([nn.Conv1d(self.hidden_size,self.cnn_nfilt, K,padding=K) for K in self.cnn_ks])
                self.decoder_cnnlayer = True
                decoder_size = self.cnn_nfilt*len(self.cnn_ks)

            self.decoder_W = nn.Parameter(torch.zeros([decoder_size, self.num_output_layers]), True)
            self.decoder_b = nn.Parameter(torch.zeros([self.num_output_layers]), True)

            if params.get('generic_classifier',False):
                self.generic_W = nn.Parameter(torch.zeros([self.hidden_size*(1+(self.max_pool_rnn==1)),1]), True)
                self.generic_b = nn.Parameter(torch.zeros([1]), True)
        else:
            self.decoder_W = nn.Parameter(torch.zeros([self.num_output_layers, self.hidden_size,
                self.output_size]), True)
            self.decoder_b = nn.Parameter(torch.zeros([self.num_output_layers, self.output_size]), True)

        #self.decoder = nn.ModuleList([nn.Linear(self.hidden_size,self.output_size) for i in
        #                             xrange(self.num_output_layers)])
        self.dec_drop = nn.Dropout(p=params.get('drop_prob_decoder',0.25))

        self.softmax = nn.Softmax()

        self.init_weights()
        # we should move it out so that whether to do cuda or not should be upto the user.
        self.cuda()

    def init_weights(self):
        # Weight initializations for various parts.
        a = np.sqrt(float(self.decoder_W.size(0)))
        if hasattr(self,'generic_W'):
            self.generic_W.data.uniform_(-a, a)
            self.generic_b.data.fill_(0.)
        if hasattr(self,'decoder_mlp'):
            n_in = np.sqrt(float(self.decoder_W_mlp.size(0)))
            self.decoder_W_mlp.data.uniform_(-1.73/n_in, 1.73/n_in)
            self.decoder_b_mlp.data.fill_(0.)
        self.decoder_W.data.uniform_(-1.73/a, 1.73/a)
        #self.encoder.weight.data.uniform_(-a, a)
        self.decoder_b.data.fill_(0)
        h_sz = self.hidden_size
        if self.compression_layer:
            n_in = np.sqrt(float(self.output_size))
            #self.compression_W.data.uniform_(0,2.*(1.73/n_in))
            qn = torch.norm(self.compression_W.weight.data, p=1, dim=1).view(-1,1).expand_as(self.compression_W.weight.data)
            self.compression_W.weight.data = self.compression_W.weight.data.div(qn)

        # LSTM forget gate could be initialized to high value (1.)
        if self.en_residual:
          for i in xrange(self.num_rec_layers):
            self.rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
            self.rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
            if self.bidir:
                self.rec_layers[i].bias_ih_l0_reverse.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
                self.rec_layers[i].bias_hh_l0_reverse.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
        else:
          for i in xrange(self.num_rec_layers):
            getattr(self.rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
            getattr(self.rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
            if self.bidir:
                getattr(self.rec_layers,'bias_ih_l'+str(i)+'_reverse').data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)
                getattr(self.rec_layers,'bias_hh_l'+str(i)+'_reverse').data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 2.)

    def init_hidden(self, bsz):
        # Weight initializations for various parts.
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_rec_layers*(1+self.bidir), bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_rec_layers*(1+self.bidir), bsz, self.hidden_size).zero_()))

    def _my_recurrent_layer(self, packed, h_prev = None):
        if self.en_residual:
            p_out = packed
            hid_all = []
            for i in xrange(self.num_rec_layers):
                if h_prev == None :
                    out, hid = self.rec_layers[i](p_out)
                else:
                    out, hid = self.rec_layers[i](p_out, (h_prev[0][i:i+1], h_prev[1][i:i+1]))
                if i > 0:
                    # Add residual connections after every layer
                    p_out = packed_add(p_out, out) if type(p_out) == torch.nn.utils.rnn.PackedSequence else p_out + out
                else:
                    p_out = out
                hid_all.append(hid)
            hidden = (torch.cat([hid[0] for hid in hid_all],dim=0), torch.cat([hid[1] for hid in hid_all],dim=0))
            rnn_out = p_out
        else:
            if h_prev == None:
                rnn_out, hidden = self.rec_layers(packed)
            else:
                rnn_out, hidden = self.rec_layers(packed, h_prev)
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

    def forward_classify(self, x, h_prev=None, compute_softmax = False, predict_mode=False, adv_inp=False, lens=None, drop=True):
        # x should be a numpy array of n_seq x n_batch dimensions
        # In this case batch will be a single sequence.
        n_auth = self.num_output_layers
        n_steps = x.size(0)
        b_sz = x.size(1)
        if not adv_inp:
            if predict_mode:
                x = Variable(x,volatile=True).cuda()
            else:
                x = Variable(x).cuda()

            if self.compression_layer:
                compressed_x = self.compression_W(x).view(n_steps*b_sz, -1)
                qn = torch.norm(compressed_x, p=1, dim=1).view(-1,1).expand_as(compressed_x) + 1e-8
                enc_x = compressed_x.div(qn).mm(self.encoder.weight).view(n_steps,b_sz, -1)
            else:
                enc_x = self.encoder(x)

            emb = self.enc_drop(enc_x) if drop else enc_x 
        else:
            if self.compression_layer:
                compressed_x = x.view(n_steps*b_sz,-1).mm(self.compression_W.weight)
                qn = torch.norm(compressed_x, p=1, dim=1).detach().view(-1,1).expand_as(compressed_x) + 1e-8
                emb_mul = compressed_x.div(qn).mm(self.encoder.weight).view(n_steps,b_sz, -1)
            else:
                emb_mul = x.view(n_steps*b_sz,-1).mm(self.encoder.weight).view(n_steps,b_sz, -1)
            #emb_mul = GradEmbMod()(x,self.encoder.weight)
            emb = self.enc_drop(emb_mul) if drop else emb_mul

        # Pack the sentences as they can be of different lens
        packed = pack_padded_sequence(emb, lens)

        rnn_out, hidden = self._my_recurrent_layer(packed, h_prev)


        if not hasattr(self, 'decoder_cnnlayer'):
            if self.max_pool_rnn==1:
                ctxt = torch.cat([packed_mean(rnn_out,dim=0), hidden[0][-1]],dim=-1)
                enc_out = self.dec_drop(ctxt) if drop else ctxt
            elif self.max_pool_rnn==2:
                rnn_unp,_= pad_packed_sequence(rnn_out)
                ctxt,_ = rnn_unp.max(dim=0)
                enc_out = self.dec_drop(ctxt) if drop else ctxt
            elif self.max_pool_rnn==3:
                rnn_unp,_= pad_packed_sequence(rnn_out)
                ctxt = torch.cat([rnn_unp.max(dim=0)[0],hidden[0].transpose(0,1).contiguous().view(b_sz,-1)],dim=-1)
                enc_out = self.dec_drop(ctxt) if drop else ctxt
            else:
                enc_out = self.dec_drop(hidden[0][-1]) if drop else hidden[0][-1]
            dec_in = enc_out

        W = self.decoder_W
        # reshape and expand b to size (n_auth*n_steps*vocab_size)
        b = self.decoder_b.expand(b_sz, self.num_output_layers)
        if hasattr(self,'decoder_mlp'):
            b_dec_mlp = self.decoder_b_mlp.expand(b_sz, self.decoder_b_mlp.size(0))
            dec_in = FN.tanh(enc_out.mm(self.decoder_W_mlp) + b_dec_mlp)
        elif hasattr(self, 'decoder_cnnlayer'):
            rnn_unpacked,_ = pad_packed_sequence(rnn_out)
            rnn_unpacked = rnn_unpacked.permute(1,2,0)
            cnn_out = [FN.leaky_relu(conv(rnn_unpacked)) for conv in self.decoder_cnn]
            cnn_pool_out = [FN.max_pool1d(cn, cn.size(2)).squeeze(2) for cn in cnn_out]
            dec_in = self.dec_drop(torch.cat(cnn_pool_out, dim=1))
            enc_out = dec_in

        dec_out = dec_in.mm(W) + b
        if compute_softmax:
            prob_out = self.softmax(dec_out.contiguous().view(-1, self.num_output_layers))
        else:
            prob_out = dec_out

        if hasattr(self,'generic_W'):
            generic_score = enc_out.mm(self.generic_W) + self.generic_b.view(1,1).expand(b_sz,1)
            generic_class = FN.sigmoid(generic_score)
            return prob_out, hidden, generic_class
        else:
            return prob_out, enc_out
