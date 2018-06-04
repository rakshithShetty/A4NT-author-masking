import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch import tensor
from tqdm import trange, tqdm
import numpy as np

class MLP_classifier(nn.Module):
    def __init__(self, params):
        super(MLP_classifier, self).__init__()
        #+1 is to allow padding index
        self.output_size = params.get('num_output_layers',205)
        self.hid_dims = params.get('hidden_widths',[])
        self.inp_size = params.get('inp_size',-1)

        prev_size = self.inp_size
        self.hid_dims.append(self.output_size)
        self.lin_layers = nn.ModuleList()
        self.non_linearities = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in xrange(len(self.hid_dims)):
            self.lin_layers.append(nn.Linear(prev_size, self.hid_dims[i]))
            self.non_linearities.append(nn.SELU())
            self.dropouts.append(nn.Dropout(p=params.get('drop_prob',0.0)))
            prev_size = self.hid_dims[i]

        self.softmax = nn.Softmax()
        self.init_weights()
        # we should move it out so that whether to do cuda or not should be upto the user.
        self.cuda()

    def init_weights(self):
        # Weight initializations for various parts.
        a = 0.01
        # LSTM forget gate could be initialized to high value (1.)
        for i in xrange(len(self.hid_dims)):
            self.lin_layers[i].weight.data.uniform_(-a, a)
            self.lin_layers[i].bias.data.fill_(0)

    def forward(self, x, compute_softmax = False):
        x = Variable(x).cuda()
        prev_out = x

        for i in xrange(len(self.hid_dims)-1):
            prev_out = self.dropouts[i](prev_out)
            prev_out = self.non_linearities[i](self.lin_layers[i](prev_out))
        prev_out = self.dropouts[-1](prev_out)
        prev_out = self.lin_layers[-1](prev_out)

        if compute_softmax:
            prob_out = self.softmax(prev_out)
        else:
            prob_out = prev_out

        return prob_out

    def fit(self, features, targs, feat_val, targ_test, epochs, lr=1e-3, l2=0.01):
        n_samples = features.shape[0]
        features = features.astype(np.float32)
        b_sz = 10
        iter_per_epoch = n_samples / b_sz
        total_iters = epochs * iter_per_epoch

        self.train()
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.RMSprop(self.parameters(), lr=lr, alpha=0.90,
                                eps=1e-8,  weight_decay=l2)
        idxes = np.arange(n_samples)
        total_loss = 0.
        #t = trange(total_iters, desc='ML')
        best_loss = 10000.

        for i in tqdm(xrange(total_iters)):
            optim.zero_grad()
            b_ids = np.random.choice(idxes, size=b_sz)
            targets = Variable(torch.from_numpy(targs[b_ids])).cuda()

            output = self.forward(torch.from_numpy(features[b_ids,:]))
            loss = criterion(output, targets)

            loss.backward()
            # Take an optimization step
            optim.step()

            total_loss += loss.data.cpu().numpy()[0]

            if i % 2000 == 0 and i > 0:
                cur_loss = total_loss / 2000.
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} |'
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    i//iter_per_epoch, i, total_iters, lr,
                    cur_loss, np.exp(cur_loss)))
                total_loss = 0.
                #if cur_loss <= best_loss:
                #    best_loss = cur_loss
                #    best_model = model.state_dict()

    def decision_function(self, features):
        n_samples = features.shape[0]
        features = features.astype(np.float32)
        b_sz = 100
        total_iters = n_samples // b_sz + 1
        self.eval()
        scores = np.zeros((n_samples, self.output_size))
        for i in tqdm(xrange(total_iters)):
            b_ids = np.arange(b_sz*i, min(n_samples,b_sz*(i+1)))
            output = self.forward(torch.from_numpy(features[b_ids,:]), compute_softmax = True)
            scores[b_ids,:] = output.data.cpu().numpy()

        return scores
