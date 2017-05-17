import argparse
import json
import time
import numpy as np
import os
from models.char_lstm import CharLstm
from collections import defaultdict
from utils.data_provider import DataProvider
from utils.utils import repackage_hidden

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math

def main(params):

    # Create vocabulary and author index
    saved_model = torch.load(params['model'])
    char_to_ix = saved_model['char_to_ix']
    auth_to_ix = saved_model['auth_to_ix']
    ix_to_char = saved_model['ix_to_char']
    cp_params = saved_model['arch']

    dp = DataProvider(cp_params)

    model = CharLstm(cp_params)
    # set to train mode, this activates dropout
    #model.test()

    # Restore saved checkpoint
    model.load_state_dict(saved_model['state_dict'])
    hidden = model.init_hidden(1)

    for i in xrange(params['num_samples']):
        batch = dp.get_random_string(slen = params['seed_length'], split=params['split'])
        inps, targs, auths, lens = dp.prepare_data(batch, char_to_ix, auth_to_ix)
        char_outs = model.forward_gen(inps, hidden, auths, n_max = params['max_len'])
        print '--------------------------------------------'
        print 'Author: %s'%(batch[0]['author'])
        print 'Seed text: %s'%(batch[0]['in'])
        print 'Gen text: %s'%(''.join([ix_to_char[c.data.cpu().tolist()[0][0]] for c in char_outs]))



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-m','--model', dest='model', type=str, default=None, help='checkpoint filename')
  parser.add_argument('-s','--split', dest='split', type=str, default='val', help='which split to evaluate')
  parser.add_argument('--num_samples', dest='num_samples', type=int, default=10, help='how many strings to generate')
  parser.add_argument('-l','--max_len', dest='max_len', type=int, default=100, help='how many characters to generate per string')
  parser.add_argument('--seed_length', dest='seed_length', type=int, default=100, help='character length of seed to the generator')
  parser.add_argument('-i', '--interactive', dest='num_eval', action='store_true', help='Should it be interactive ')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
