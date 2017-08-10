from tqdm import tqdm
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


def adv_eval_pass(modelGen, modelEval, inps, lens, end_c=0, maxlen=100, auths=None):

    char_outs = modelGen.forward_gen(inps, end_c=end_c, n_max=maxlen, auths=auths)
    #--------------------------------------------------------------------------
    # The output need to be sorted by length to be fed into further LSTM stages
    #--------------------------------------------------------------------------
    gen_len = len(char_outs)
    eval_inp = torch.unsqueeze(torch.cat(char_outs),1).data
    if (gen_len <= 0):
        import ipdb
        ipdb.set_trace()


    #---------------------------------------------------
    # Now pass the generated samples to the evaluator
    # output has format: [auth_classifier out, hidden state, generic classifier out (optional])
    #---------------------------------------------------
    eval_out_gen = modelEval.forward_classify(eval_inp, lens=[gen_len], compute_softmax=True)
    # Undo the sorting here
    samples_out = (gen_len, char_outs)

    return eval_out_gen + samples_out

def main(params):

    # Create vocabulary and author index
    saved_model = torch.load(params['genmodel'])
    cp_params = saved_model['arch']
    if params['evalmodel']:
        eval_model = torch.load(params['evalmodel'])
        eval_params = eval_model['arch']
        eval_state = eval_model['state_dict']
    else:
        print "FIX THIS"
        return

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

    dp = DataProvider(cp_params)
    modelGen = CharTranslator(cp_params)
    modelEval = CharLstm(eval_params)

    modelGen.eval()
    modelEval.eval()

    # Restore saved checkpoint
    modelGen.load_state_dict(saved_model['state_dict'])
    state = modelEval.state_dict()
    state.update(eval_state)
    modelEval.load_state_dict(state)

    accum_diff_eval = [[],[]]
    accum_err_eval = np.zeros(len(auth_to_ix))
    accum_err_real = np.zeros(len(auth_to_ix))
    accum_count_gen = np.zeros(len(auth_to_ix))


    jc = '' if cp_params.get('atoms','char') == 'char' else ' '

    for i, b_data in tqdm(enumerate(dp.iter_sentences(split=params['split'], atoms=cp_params.get('atoms','word'), batch_size = 1))):
        if i > params['num_samples'] and params['num_samples']>0:
            break;
    #for i in xrange(params['num_samples']):
        #c_aid = np.random.choice(auth_to_ix.values())
        #batch = dp.get_sentence_batch(1,split=params['split'], atoms=cp_params.get('atoms','char'), aid=ix_to_auth[c_aid])
        inps, targs, auths, lens = dp.prepare_data(b_data[0], char_to_ix, auth_to_ix)
        # outs are organized as
        auths_inp = 1 - auths if params['flip'] else auths
        outs = adv_eval_pass(modelGen, modelEval, inps, lens, end_c=char_to_ix['.'], maxlen=cp_params['max_seq_len'], auths=auths_inp)

        eval_out_gt = modelEval.forward_classify(targs, lens=lens, compute_softmax=True)
        real_aid_out = eval_out_gt[0][:, auths_inp[0]]

        gen_aid_out = outs[0][:, auths_inp[0]]
        accum_err_eval[auths_inp[0]] += (gen_aid_out.data >= 0.5).float().mean()
        accum_err_real[auths_inp[0]] += (real_aid_out.data >= 0.5).float().mean()
        accum_count_gen[auths_inp[0]] += 1.
        accum_diff_eval[auths_inp[0]].append(gen_aid_out.data[0] - real_aid_out.data[0])
        if params['print']:
            print '--------------------------------------------'
            print 'Author: %s'%(b_data[0][0]['author'])
            print 'Inp text %s: %s (%.2f)'%(ix_to_auth[auths[0]], jc.join([c for c in b_data[0][0]['in'] if c in char_to_ix]), real_aid_out.data[0])
            print 'Out text %s: %s (%.2f)'%(ix_to_auth[auths_inp[0]],jc.join([ix_to_char[c.data.cpu()[0]] for c in outs[-1]]), gen_aid_out.data[0])
        #else:
        #    print '%d/%d\r'%(i, params['num_samples']),

    err_a1, err_a2 = accum_err_eval[0]/(1e-5+accum_count_gen[0]), accum_err_eval[1]/(1e-5+accum_count_gen[1])
    err_real_a1, err_real_a2 = accum_err_real[0]/(1e-5+accum_count_gen[0]), accum_err_real[1]/(1e-5+accum_count_gen[1])
    print(' erra1 {:3.2f} - erra2 {:3.2f}'.format(100.*err_a1, 100.*err_a2))
    print(' err_real_a1 {:3.2f} - err_real_a2 {:3.2f}'.format(100.*err_real_a1, 100.*err_real_a2))
    print(' count %d - %d'%(accum_count_gen[0], accum_count_gen[1]))

    diff_arr0, diff_arr1 =  np.array(accum_diff_eval[0]), np.array(accum_diff_eval[1])
    print 'Mean difference : translation to %s = %.2f , translation to %s = %.2f '%(ix_to_auth[0], diff_arr0.mean(), ix_to_auth[1], diff_arr1.mean())
    print 'Difference > 0  : translation to %s = %.2f%%, translation to %s = %.2f%% '%(ix_to_auth[0], 100.*(diff_arr0>0).sum()/(1e-5+diff_arr0.shape[0]), ix_to_auth[1], 100.*(diff_arr1>0).sum()/(1e-5+diff_arr1.shape[0]))



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-g','--genmodel', dest='genmodel', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-e','--evalmodel', dest='evalmodel', type=str, default=None, help='evakcheckpoint filename')
  parser.add_argument('-s','--split', dest='split', type=str, default='val', help='which split to evaluate')
  parser.add_argument('--num_samples', dest='num_samples', type=int, default=0, help='how many strings to generate')
  parser.add_argument('-l','--max_len', dest='max_len', type=int, default=100, help='how many characters to generate per string')
  parser.add_argument('--seed_length', dest='seed_length', type=int, default=100, help='character length of seed to the generator')
  parser.add_argument('-i', '--interactive', dest='interactive', action='store_true', help='Should it be interactive ')
  parser.add_argument('--m_type', dest='m_type', type=str, default='generative', help='type')
  parser.add_argument('--flip', dest='flip', type=int, default=0, help='flip authors')
  parser.add_argument('--print', dest='print', type=int, default=0, help='Print scores')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)

