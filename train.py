import argparse
import json
import time
import numpy as np
import os
from models.char-lstm import CharLstm
from collections import defaultdict
from utils.data_provider import DataProvider

def main(params):

    #TODO
    dp = DataProvider(params)
    ntokens = len(corpus.dictionary)

    model = CharLstm(params)

    # Let's now train the model
    # set to train mode
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(params['batch_size'])

    for batch, i in enumerate(range(0, train_data.size(0) - 1, parms['max_seq_len'])):
        #TODO
        data, targets = dp.get_doc_batch(split='train')
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        #TODO
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', default='coco', help='dataset: pan')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=50, help='maximum sequence length')
  parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='max batch size')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
