import argparse
import json
import time
import numpy as np
import os
from models.char_lstm import CharLstm
from collections import defaultdict
from utils.data_provider import DataProvider
from utils.utils import repackage_hidden, eval_model
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math

def save_checkpoint(state, fappend ='dummy', outdir = 'cv'):
    filename = os.path.join(outdir,'checkpoint_'+fappend+'_'+str(state['val_auc'])+'.pth.tar')
    torch.save(state, filename)

def main(params):

    dp = DataProvider(params)

    # Create vocabulary and author index
    if params['resume'] == None:
        char_to_ix, ix_to_char = dp.createVocab(params['vocab_threshold'])
        auth_to_ix = dp.createAuthorIdx()
    else:
        saved_model = torch.load(params['resume'])
        char_to_ix = saved_model['char_to_ix']
        auth_to_ix = saved_model['auth_to_ix']
        ix_to_char = saved_model['ix_to_char']

    params['vocabulary_size'] = len(char_to_ix)
    params['num_output_layers'] = len(auth_to_ix)

    model = CharLstm(params)
    # set to train mode, this activates dropout
    model.train()
    #Initialize the RMSprop optimizer
    optim = torch.optim.RMSprop(model.parameters(), lr=params['learning_rate'], alpha=params['decay_rate'],
                                eps=params['smooth_eps'])
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Restore saved checkpoint
    if params['resume'] !=None:
        model.load_state_dict(saved_model['state_dict'])
        optim.load_state_dict(saved_model['optimizer'])

    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(params['batch_size'])

    # Compute the iteration parameters
    epochs = params['max_epochs']
    total_seqs = dp.get_num_seqs(maxlen=params['max_seq_len'], split='train')
    iter_per_epoch = total_seqs // params['batch_size']
    total_iters = iter_per_epoch * epochs
    best_loss = 1000000.
    best_val = 0.
    eval_every = int(iter_per_epoch * params['eval_interval'])

    #val_score = eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])
    val_score = 0. #eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])
    val_rank = 1000

    for i in xrange(total_iters):
        #TODO
        batch, reset_h = dp.get_doc_batch(split='train')
        inps, targs, auths, lens = dp.prepare_data(batch, char_to_ix, auth_to_ix, leakage=params['leakage'])

        # Reset the hidden states for which new docs have been sampled

        if len(reset_h) > 0:
            hidden[0].data.index_fill_(1,torch.LongTensor(reset_h).cuda(),0.)
            hidden[1].data.index_fill_(1,torch.LongTensor(reset_h).cuda(),0.)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optim.zero_grad()

        output, hidden = model.forward(inps, lens, hidden, auths)
        #TODO
        targets = pack_padded_sequence(Variable(targs).cuda(),lens)
        loss = criterion(pack_padded_sequence(output,lens)[0], targets[0])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), params['grad_clip'])

        # Take an optimization step
        optim.step()

        total_loss += loss.data.cpu().numpy()[0]

        if i % eval_every == 0 and i > 0:
            val_rank, val_score = eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])

        if i % params['log_interval'] == 0 and i > 0:
            cur_loss = total_loss / params['log_interval']
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                i//iter_per_epoch, i, total_iters, params['learning_rate'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.

            if val_score >=best_val:
                save_checkpoint({
                    'iter': i,
                    'arch': params,
                    'val_mean_rank': val_rank,
                    'val_auc': val_score,
                    'char_to_ix': char_to_ix,
                    'ix_to_char': ix_to_char,
                    'auth_to_ix': auth_to_ix,
                    'state_dict': model.state_dict(),
                    'loss':  cur_loss,
                    'optimizer' : optim.state_dict(),
                }, fappend = params['fappend'],
                outdir = params['checkpoint_output_directory'])
                best_val = val_score
            start_time = time.time()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', default='pan16AuthorMask', help='dataset: pan')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--resume', dest='resume', type=str, default=None, help='append this string to checkpoint filenames')
  parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=50, help='maximum sequence length')
  parser.add_argument('--vocab_threshold', dest='vocab_threshold', type=int, default=5, help='vocab threshold')

  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=10, help='max batch size')

  # Optimization parameters
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.99, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=50, help='number of epochs to train for')

  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')

  parser.add_argument('--leakage', dest='leakage', type=float, default=1e-3, help='decay rate for adadelta/rmsprop')


  # Validation args
  parser.add_argument('--eval', dest='eval_interval', type=float, default=0.5, help='print every x iters')
  parser.add_argument('--num_eval', dest='num_eval', type=int, default=-1, help='print every x iters')
  parser.add_argument('--log', dest='log_interval', type=int, default=1, help='print every x iters')

  # LSTM parameters
  parser.add_argument('--en_residual_conn', dest='en_residual_conn', type=int, default=0, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--embedding_size', dest='embedding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
