import argparse
import json
import time
import numpy as np
import os
from models.model_utils import get_classifier
from models.char_lstm import CharLstm
from models.char_translator import CharTranslator
from collections import defaultdict
from utils.data_provider import DataProvider
from utils.utils import repackage_hidden, eval_model, eval_classify, eval_translator

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

    if params['m_type'] == 'translator':
        model = CharTranslator(cp_params)
    else:
        model = get_classifier(cp_params)
    # set to train mode, this activates dropout
    #model.eval()

    # Restore saved checkpoint
    model.load_state_dict(saved_model['state_dict'])

    eval_function = eval_translator if params['m_type']=='translator' else eval_model if cp_params['mode'] == 'generative' else eval_classify

    score = eval_function(dp, model, cp_params, char_to_ix, auth_to_ix, split=params['split'], max_docs = params['num_eval'], dump_scores=params['dump_scores'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-m','--model', dest='model', type=str, default=None, help='checkpoint filename')
  parser.add_argument('--m_type', dest='m_type', type=str, default='generator', help='checkpoint filename')
  parser.add_argument('-s','--split', dest='split', type=str, default=None, help='which split to evaluate')
  parser.add_argument('--num_eval', dest='num_eval', type=int, default=-1, help='how many doc to evlauate')
  parser.add_argument('--dump_scores', dest='dump_scores', type=int, default=0, help='how many doc to evlauate')
  parser.add_argument('--topk', dest='topk', type=int, default=5, help='how many doc to evlauate')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
