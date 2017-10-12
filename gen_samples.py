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
from torch.autograd import Variable, Function

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import math
from termcolor import colored

def adv_forward_pass(modelGen, inps, lens, end_c=0, maxlen=100, auths=None,
        cycle_compute=True, append_symb=None):
    modelGen.eval()

    char_outs = modelGen.forward_gen(inps, end_c=end_c, n_max=maxlen, auths=auths)
    gen_len = len(char_outs)
    #--------------------------------------------------------------------------
    # The output need to be sorted by length to be fed into further LSTM stages
    #--------------------------------------------------------------------------
    eval_inp = torch.unsqueeze(torch.cat(char_outs),1).data
    #---------------------------------------------------
    # Now pass the generated samples to the evaluator
    # output has format: [auth_classifier out, hidden state, generic classifier out (optional])
    #---------------------------------------------------
    if cycle_compute:
        reverse_inp = torch.cat([append_symb, eval_inp],dim=0)
        #reverse_inp = reverse_inp.detach()
        rev_char_outs = modelGen.forward_gen(reverse_inp, end_c=end_c, n_max=maxlen, auths=1-auths)
        samples_out = (char_outs, rev_char_outs)
    else:
        samples_out = (char_outs, )

    return samples_out

#def adv_eval_pass(modelGen, modelEval, inps, lens, end_c=0, maxlen=100, auths=None):
#
#    char_outs = modelGen.forward_gen(inps, end_c=end_c, n_max=maxlen, auths=auths)
#    #--------------------------------------------------------------------------
#    # The output need to be sorted by length to be fed into further LSTM stages
#    #--------------------------------------------------------------------------
#    gen_len = len(char_outs)
#    eval_inp = torch.unsqueeze(torch.cat(char_outs),1).data
#    if (gen_len <= 0):
#        import ipdb
#        ipdb.set_trace()
#
#    #---------------------------------------------------
#    # Now pass the generated samples to the evaluator
#    # output has format: [auth_classifier out, hidden state, generic classifier out (optional])
#    #---------------------------------------------------
#    eval_out_gen = modelEval.forward_classify(eval_inp, lens=[gen_len], compute_softmax=True)
#    # Undo the sorting here
#    samples_out = (gen_len, char_outs)

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
    if params['softmax_scale']:
        cp_params['softmax_scale'] = params['softmax_scale']

    dp = DataProvider(cp_params)

    if params['m_type'] == 'generative':
        model = CharLstm(cp_params)
    else:
        model = CharTranslator(cp_params)
    # set to train mode, this activates dropout
    model.eval()
    auth_colors = ['red', 'blue']

    startc = dp.data['configs']['start']
    endc = dp.data['configs']['end']

    append_tensor = np.zeros((1, 1), dtype=np.int)
    append_tensor[0, 0] = char_to_ix[startc]
    append_tensor = torch.LongTensor(append_tensor).cuda()

    # Restore saved checkpoint
    model.load_state_dict(saved_model['state_dict'])
    hidden = model.init_hidden(1)
    jc = '' if cp_params.get('atoms','char') == 'char' else ' '

    for i in xrange(params['num_samples']):
        c_aid = np.random.choice(auth_to_ix.values())
        if params['m_type'] == 'generative':
            batch = dp.get_random_string(slen = params['seed_length'], split=params['split'])
        else:
            batch = dp.get_sentence_batch(1,split=params['split'], atoms=cp_params.get('atoms','char'), aid=ix_to_auth[c_aid])

        inps, targs, auths, lens = dp.prepare_data(batch, char_to_ix, auth_to_ix, maxlen=cp_params['max_seq_len'])
        auths_inp = 1 - auths if params['flip'] else auths
        outs = adv_forward_pass(model, inps, lens, end_c=char_to_ix[endc], maxlen=cp_params['max_seq_len'], auths=auths_inp,
                     cycle_compute=params['show_rev'], append_symb=append_tensor)
        #char_outs = model.forward_gen(inps, hidden, auths_inp, n_max = cp_params['max_len'],end_c=char_to_ix['.'])
        print '--------------------------------------------'
        #print 'Translate from %s to %s'%(batch[0]['author'], ix_to_auth[auths_inp[0]])
        print colored('Inp %6s: '%(ix_to_auth[auths[0]]),'green') + colored('%s'%(jc.join([ix_to_char[c[0]] for c in inps[1:]])),auth_colors[auths[0]])
        print colored('Out %6s: '%(ix_to_auth[auths_inp[0]]),'grey')+ colored('%s'%(jc.join([ix_to_char[c.data.cpu()[0]] for c in outs[0] if c.data.cpu()[0] in ix_to_char])),auth_colors[auths_inp[0]])

        if params['show_rev']:
            print colored('Rev %6s: '%(ix_to_auth[auths[0]]),'green')+ colored('%s'%(jc.join([ix_to_char[c.data.cpu()[0]] for c in outs[-1] if c.data.cpu()[0] in ix_to_char])),auth_colors[auths[0]])



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-m','--model', dest='model', type=str, default=None, help='checkpoint filename')
  parser.add_argument('-s','--split', dest='split', type=str, default='val', help='which split to evaluate')
  parser.add_argument('--num_samples', dest='num_samples', type=int, default=10, help='how many strings to generate')
  parser.add_argument('--show_rev', dest='show_rev', type=int, default=0, help='how many strings to generate')
  parser.add_argument('-l','--max_len', dest='max_len', type=int, default=100, help='how many characters to generate per string')
  parser.add_argument('--seed_length', dest='seed_length', type=int, default=100, help='character length of seed to the generator')
  parser.add_argument('-i', '--interactive', dest='interactive', action='store_true', help='Should it be interactive ')
  parser.add_argument('--m_type', dest='m_type', type=str, default='generative', help='type')
  parser.add_argument('--flip', dest='flip', type=int, default=0, help='flip authors')

  parser.add_argument('--softmax_scale', dest='softmax_scale', type=int, default=None, help='how many samples per sentence')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
