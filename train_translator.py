import argparse
import json
import time
import numpy as np
import os
from models.char_lstm import CharLstm
from models.char_translator import CharTranslator
from collections import defaultdict
from utils.data_provider import DataProvider
from utils.utils import repackage_hidden, eval_translator, eval_classify
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math


def nll_loss(outputs, targets):
    return torch.gather(outputs, 1, targets.view(-1,1))

def save_checkpoint(state, fappend ='dummy', outdir = 'cv'):
    filename = os.path.join(outdir,'checkpoint_translate_'+fappend+'_'+'%.2f'%(state['val_pplx'])+'.pth.tar')
    torch.save(state, filename)

def main(params):

    dp = DataProvider(params)

    # Create vocabulary and author index
    if params['resume'] == None:
        if params['atoms'] == 'char':
            char_to_ix, ix_to_char = dp.createCharVocab(params['vocab_threshold'])
        else:
            char_to_ix, ix_to_char = dp.createWordVocab(params['vocab_threshold'])
        auth_to_ix, ix_to_auth = dp.createAuthorIdx()
    else:
        saved_model = torch.load(params['resume'])
        char_to_ix = saved_model['char_to_ix']
        auth_to_ix = saved_model['auth_to_ix']
        ix_to_auth = saved_model['ix_to_auth']
        ix_to_char = saved_model['ix_to_char']

    params['vocabulary_size'] = len(char_to_ix)
    params['num_output_layers'] = len(auth_to_ix)

    model = CharTranslator(params)
    # set to train mode, this activates dropout
    model.train()
    #Initialize the RMSprop optimizer

    if params['use_sgd']:
        optim = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum = params['decay_rate'])
    else:
        optim = torch.optim.RMSprop(model.parameters(), lr=params['learning_rate'], alpha=params['decay_rate'],
                                eps=params['smooth_eps'])
    # Loss function
    if params['mode'] == 'generative':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.NLLLoss()

    # Restore saved checkpoint
    if params['resume'] !=None:
        model.load_state_dict(saved_model['state_dict'])
        optim.load_state_dict(saved_model['optimizer'])

    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(params['batch_size'])
    hidden_zeros = model.init_hidden(params['batch_size'])
    # Initialize the cache
    if params['randomize_batches']:
        dp.set_hid_cache(range(len(dp.data['docs'])), hidden_zeros)

    # Compute the iteration parameters
    epochs = params['max_epochs']
    total_seqs = dp.get_num_sents(split='train')
    iter_per_epoch = total_seqs // params['batch_size']
    total_iters = iter_per_epoch * epochs
    best_loss = 1000000.
    best_val = 1000.
    eval_every = int(iter_per_epoch * params['eval_interval'])

    #val_score = eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])
    val_score = 0. #eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])
    val_rank = 1000

    eval_function = eval_translator if params['mode'] == 'generative' else eval_classify
    leakage = 0. #params['leakage']

    print total_iters
    for i in xrange(total_iters):
        #TODO
        if params['split_generators']:
            c_aid = ix_to_auth[np.random.choice(auth_to_ix.values())]
        else:
            c_aid = None

        batch = dp.get_sentence_batch(params['batch_size'], split='train',
                    atoms=params['atoms'], aid=c_aid, sample_by_len = params['sample_by_len'])
        inps, targs, auths, lens = dp.prepare_data(batch, char_to_ix, auth_to_ix, maxlen=params['max_seq_len'])
        # Reset the hidden states for which new docs have been sampled

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optim.zero_grad()
        #TODO
        if params['mode'] == 'generative':
            output, _ = model.forward_mltrain(inps, lens, inps, lens, hidden_zeros, auths=auths)
            targets = pack_padded_sequence(Variable(targs).cuda(),lens)
            loss = criterion(pack_padded_sequence(output,lens)[0], targets[0])
        else:
            # for classifier auths is the target
            output, hidden = model.forward_classify(inps, hidden, compute_softmax=True)
            targets = Variable(auths).cuda()
            loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), params['grad_clip'])

        # Take an optimization step
        optim.step()

        total_loss += loss.data.cpu().numpy()[0]

        # Save the hidden states in cache for later use
        if i % eval_every == 0 and i > 0:
            val_rank, val_score = eval_function(dp, model, params, char_to_ix, auth_to_ix, split='val')

        #if i % iter_per_epoch == 0 and i > 0 and leakage > params['leakage_min']:
        #    leakage = leakage * params['leakage_decay']

        #if (i % iter_per_epoch == 0) and ((i//iter_per_epoch) >= params['lr_decay_st']):
        if i % params['log_interval'] == 0 and i > 0:
            cur_loss = total_loss / params['log_interval']
            elapsed = time.time() - start_time
            print('| epoch {:2.2f} | {:5d}/{:5d} batches | lr {:02.2e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                float(i)/iter_per_epoch, i, total_iters, params['learning_rate'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.

            if val_rank <=best_val:
                save_checkpoint({
                    'iter': i,
                    'arch': params,
                    'val_loss': val_rank,
                    'val_pplx': val_score,
                    'char_to_ix': char_to_ix,
                    'ix_to_char': ix_to_char,
                    'auth_to_ix': auth_to_ix,
                    'ix_to_auth': ix_to_auth,
                    'state_dict': model.state_dict(),
                    'loss':  cur_loss,
                    'optimizer' : optim.state_dict(),
                }, fappend = params['fappend'],
                outdir = params['checkpoint_output_directory'])
                best_val = val_rank
            start_time = time.time()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', default='pan16AuthorMask', help='dataset: pan')
  parser.add_argument('--datasetfile', dest='dataset_file', default='dataset.json', help='dataset: pan')
  # dataset options
  parser.add_argument('--authstring', dest='authstring', type=str, default='author', help='which label to use as author')
  parser.add_argument('--filterauths', dest='filterauths', nargs='+', type=str, default=[], help='which author classes to keep')
  parser.add_argument('--filtertype', dest='filtertype', type=str, default='keep', help='which author classes to keep')

  parser.add_argument('--use_unk', dest='use_unk', type=int, default=0, help='Use UNK for out of vocabulary words')
  parser.add_argument('--uniform_len_sample', dest='uniform_len_sample', type=int, default=0, help='uniform_len_sample')
  parser.add_argument('--sample_by_len', dest='sample_by_len', type=int, default=1, help='Use UNK for out of vocabulary words')

  # mode
  parser.add_argument('--mode', dest='mode', type=str, default='generative', help='print every x iters')
  parser.add_argument('--atoms', dest='atoms', type=str, default='char', help='character or word model')
  parser.add_argument('--maxpoolrnn', dest='maxpoolrnn', type=int, default=0, help='maximum sequence length')
  parser.add_argument('--pad_auth_vec', dest='pad_auth_vec', type=int, default=10, help='maximum sequence length')

  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--resume', dest='resume', type=str, default=None, help='append this string to checkpoint filenames')
  parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=50, help='maximum sequence length')
  parser.add_argument('--vocab_threshold', dest='vocab_threshold', type=int, default=5, help='vocab threshold')

  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=10, help='max batch size')
  parser.add_argument('--randomize_batches', dest='randomize_batches', type=int, default=1, help='randomize batches')

  # Optimization parameters
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.95, help='solver learning rate')
  parser.add_argument('--lr_decay_st', dest='lr_decay_st', type=int, default=0, help='solver learning rate')

  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.99, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--use_sgd', dest='use_sgd', type=int, default=0, help='Use sgd')
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=50, help='number of epochs to train for')

  parser.add_argument('--drop_prob_emb', dest='drop_prob_emb', type=float, default=0.25, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.25, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.25, help='what dropout to apply right before the decoder in an RNN/LSTM')

  # Validation args
  parser.add_argument('--eval_interval', dest='eval_interval', type=float, default=0.5, help='print every x iters')
  parser.add_argument('--num_eval', dest='num_eval', type=int, default=-1, help='print every x iters')
  parser.add_argument('--log', dest='log_interval', type=int, default=1, help='print every x iters')

  # Add noise instead of dropout after encoder
  parser.add_argument('--apply_noise', dest='apply_noise', type=int, default=None)
  
  # No encoder network 
  parser.add_argument('--no_encoder', dest='no_encoder', type=int, default=0)


  # LSTM parameters
  parser.add_argument('--en_residual_conn', dest='en_residual_conn', type=int, default=0, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--split_generators', dest='split_generators',
                        type=int, default=0, help='Split the generators')

  parser.add_argument('--embedding_size', dest='embedding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--enc_hidden_depth', dest='enc_hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--enc_hidden_size', dest='enc_hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--dec_hidden_depth', dest='dec_hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--dec_hidden_size', dest='dec_hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print json.dumps(params, indent = 2)
  main(params)
