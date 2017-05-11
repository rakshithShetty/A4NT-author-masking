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
        self.enc_drop = nn.Dropout(p=params.get('enc_drop_prob',0.25))

        # Lstm Layers
        if self.en_residual:
            self.rec_layers = nn.ModuleList([nn.LSTM(self.emb_size, self.hidden_size, 1) for i in xrange(self.num_rec_layers)])
        else:
            self.rec_layers = nn.LSTM(self.emb_size, self.hidden_size, self.num_rec_layers)

        # Output decoder layer
        self.decoder_W = nn.Parameter(torch.zeros([self.num_output_layers, self.hidden_size,
            self.output_size]), True)
        self.decoder_b = nn.Parameter(torch.zeros([self.num_output_layers, self.output_size]), True)

        #self.decoder = nn.ModuleList([nn.Linear(self.hidden_size,self.output_size) for i in
        #                             xrange(self.num_output_layers)])
        self.dec_drop = nn.Dropout(p=params.get('dec_drop_prob',0.25))

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
            self.rec_layers[i].bias_ih_l0.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 1.)
            self.rec_layers[i].bias_hh_l0.data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 1.)
        else:
          for i in xrange(self.num_rec_layers):
            getattr(self.rec_layers,'bias_ih_l'+str(i)).data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 1.)
            getattr(self.rec_layers,'bias_hh_l'+str(i)).data.index_fill_(0, torch.arange(h_sz +1, h_sz*2).long(), 1.)

    def init_hidden(self, bsz):
        # Weight initializations for various parts.
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_rec_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_rec_layers, bsz, self.hidden_size).zero_()))


    def forward(self, x, lengths, h_prev, target_head, compute_softmax = False):
        # x should be a numpy array of n_seq x n_batch dimensions
        b_sz = x.size(1)
        n_steps = x.size(0)
        x = Variable(x).cuda()
        emb = self.enc_drop(self.encoder(x))
        packed = pack_padded_sequence(emb, lengths)

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
            rnn_out, hidden = self.rec_layers(packed, h_prev)

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
