import argparse
import json
import time
import numpy as np
import os
from models.char_lstm import CharLstm
from models.char_translator import CharTranslator
from collections import defaultdict
from utils.data_provider import DataProvider
from utils.utils import repackage_hidden

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math
from termcolor import colored

def main(params):

    # Create vocabulary and author index
    saved_model = torch.load(params['model'])
    if 'misc' in saved_model:
        misc = saved_model['misc']
        char_to_ix = misc['char_to_ix']
        auth_to_ix = misc['auth_to_ix']
        ix_to_char = misc['ix_to_char']
        ix_to_auth = misc['ix_to_auth']
    else:
        char_to_ix = saved_model['char_to_ix']
        auth_to_ix = saved_model['auth_to_ix']
        ix_to_char = saved_model['ix_to_char']
        ix_to_auth = saved_model['ix_to_auth']
    cp_params = saved_model['arch']

    dp = DataProvider(cp_params)

    if params['m_type'] == 'generative':
        model = CharLstm(cp_params)
    else:
        model = CharTranslator(cp_params)
    # set to train mode, this activates dropout
    model.eval()
    auth_colors = ['red', 'blue']

    # Restore saved checkpoint
    model.load_state_dict(saved_model['state_dict'])
    hidden = model.init_hidden(1)
    jc = '' if cp_params.get('atoms','char') == 'char' else ' '

    for i in xrange(params['num_samples']):
        if params['m_type'] == 'generative':
            batch = dp.get_random_string(slen = params['seed_length'], split=params['split'])
        else:
            batch = dp.get_sentence_batch(1,split=params['split'], atoms=cp_params.get('atoms','char'))

        inps, targs, auths, lens = dp.prepare_data(batch, char_to_ix, auth_to_ix)
        auths_inp = 1 - auths if params['flip'] else auths
        char_outs = model.forward_gen(inps, hidden, auths_inp, n_max = params['max_len'],end_c=char_to_ix['.'])
        print '--------------------------------------------'
        #print 'Translate from %s to %s'%(batch[0]['author'], ix_to_auth[auths_inp[0]])
        print colored('Inp %s: '%(ix_to_auth[auths[0]]),'green') + colored('%s'%(jc.join([c for c in batch[0]['in'][1:] if c in char_to_ix])),auth_colors[auths[0]])
        print colored('Out %s: '%(ix_to_auth[auths_inp[0]]),'grey')+ colored('%s'%(jc.join([ix_to_char[c.data.cpu()[0]] for c in char_outs])),auth_colors[auths_inp[0]])



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-m','--model', dest='model', type=str, default=None, help='checkpoint filename')
  parser.add_argument('-s','--split', dest='split', type=str, default='val', help='which split to evaluate')
  parser.add_argument('--num_samples', dest='num_samples', type=int, default=10, help='how many strings to generate')
  parser.add_argument('-l','--max_len', dest='max_len', type=int, default=100, help='how many characters to generate per string')
  parser.add_argument('--seed_length', dest='seed_length', type=int, default=100, help='character length of seed to the generator')
  parser.add_argument('-i', '--interactive', dest='interactive', action='store_true', help='Should it be interactive ')
  parser.add_argument('--m_type', dest='m_type', type=str, default='generative', help='type')
  parser.add_argument('--flip', dest='flip', type=int, default=0, help='flip authors')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
