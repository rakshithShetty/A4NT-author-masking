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
from pycrayon import CrayonClient
import time

def calc_gradient_penalty(netD, real_data, real_lens, fake_data, fake_lens, targs, endc=0):
    #print real_data.size()
    b_sz = real_data.shape[1]
    import ipdb; ipdb.set_trace()
    alpha = np.random.rand()
    if real_data.shape[0] > fake_data.shape[0]:
        n_samp = real_data.shape[0]
        fake_data = np.concatenate([fake_data,np.zeros((n_samp-fake_data.shape[0],b_sz))], dim=0)
    elif real_data.shape[0] < fake_data.shape[0]:
        n_samp = fake_data.shape[0]
        real_data = np.concatenate([real_data,np.zeros((n_samp-real_data.shape[0],b_sz))], dim=0)

    interp_lens = (real_lens * alpha + fake_lens * (1-alpha)).round().astype(int)
    real_valid_mask = (np.tile(np.arange(n_samp)[:,None],[1,b_sz]) < np.minimum(real_lens,interp_lens-1))
    fake_valid_mask = (np.tile(np.arange(n_samp)[:,None],[1,b_sz]) < np.minimum(fake_lens,interp_lens-1))

    beta = np.random.binomial(1,alpha,(n_samp,b_sz))
    # This interpolation doesn't make much sense
    interpolates = ((real_valid_mask & (beta|~fake_valid_mask))*real_data) + ((fake_valid_mask
            &((1-beta)|~real_valid_mask))* fake_data)
    interpolates[interp_lens-1,np.arange(b_sz)] = endc


    eval_out_interp, _ = modelEval.forward_classify(interpolates, lens=interp_lens.tolist())
    gradients = autograd.grad(outputs=eval_out_interp, inputs=interpolates,
                              grad_outputs=torch.ones(eval_out_interp.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def nll_loss(outputs, targets):
    return torch.gather(outputs, 1, targets.view(-1,1))

def save_checkpoint(state, fappend ='dummy', outdir = 'cv'):
    filename = os.path.join(outdir,'checkpoint_gan_'+fappend+'_'+'%.2f'%(state['val_pplx'])+'.pth.tar')
    torch.save(state, filename)

def disp_gen_samples(modelGen, dp, misc, maxlen=100, n_disp = 5, atoms='char'):
    modelGen.eval()
    ix_to_char = misc['ix_to_char']
    jc = '' if atoms == 'char' else ' '
    batch = dp.get_sentence_batch(n_disp, split='train', atoms=atoms)
    inps, targs, auths, lens = dp.prepare_data(batch, misc['char_to_ix'], misc['auth_to_ix'], maxlen=maxlen)
    gen_samples, gen_lens, char_outs = modelGen.forward_advers_gen(inps, lens,
                    soft_samples=True, end_c=misc['char_to_ix']['.'],
                    n_max=maxlen, auths=1-auths)
    print '----------------------Visualising Some Generated Samples-----------------------------------------\n'
    for i in xrange(len(lens)):
        print '%d Inp : %s --> %s'%(i, misc['ix_to_auth'][auths[i]], jc.join([ix_to_char[c] for c in inps.numpy()[1:,i] if c in ix_to_char]))
        print '%d Out : %s --> %s'%(i, misc['ix_to_auth'][auths[i]], jc.join([ix_to_char[c[i]] for c in char_outs[:gen_lens[i]]]))
    print '\n-------------------------------------------------------------------------------------------------'
    modelGen.train()

def adv_forward_pass(modelGen, modelEval, inps, evalInps, lens, end_c=0, backprop_for='all', maxlen=100, auths=None,
        cycle_compute= False, append_symb=None):
    gen_samples, gen_lens, char_outs = modelGen.forward_advers_gen(inps, lens, soft_samples=True, end_c=end_c,
            n_max=maxlen, auths=1-auths)
    if (gen_lens <=0).any():
        import ipdb; ipdb.set_trace()

    #--------------------------------------------------------------------------
    # The output need to be sorted by length to be fed into further LSTM stages
    #--------------------------------------------------------------------------
    len_sorted, gen_lensort_idx = gen_lens.sort(dim=0,descending=True)
    _, rev_sort_idx = gen_lensort_idx.sort(dim=0)
    rev_sort_idx = Variable(rev_sort_idx, requires_grad=False)

    #--------------------------------------------------------------------------
    gen_samples = torch.cat([torch.unsqueeze(gs,0) for gs in gen_samples],dim=0)
    gen_samples = gen_samples.index_select(1, Variable(gen_lensort_idx,requires_grad=False))
    if backprop_for == 'eval':
        gen_samples = gen_samples.detach()

    #---------------------------------------------------
    # Now pass the generated samples to the evaluator
    # output has format: [auth_classifier out, hidden state, generic classifier out (optional])
    #---------------------------------------------------
    eval_out_gen = modelEval.forward_classify(gen_samples, adv_inp=True, lens=len_sorted.tolist())
    # Undo the sorting here
    eval_out_gen_sort = eval_out_gen[0].index_select(0,rev_sort_idx)

    #Pass GT samples to evaluator
    if backprop_for == 'eval':
        #output has format: [auth_classifier out, hidden state, generic classifier out (optional)]
        eval_out_gt = modelEval.forward_classify(evalInps, lens=lens)
    else:
        eval_out_gt = ()

    if cycle_compute and backprop_for != 'eval':
        reverse_inp = torch.cat([append_symb,gen_samples[:-1,:,:]])
        #reverse_inp = reverse_inp.detach()
        rev_gen_samples, rev_gen_lens, rev_char_outs = modelGen.forward_advers_gen(reverse_inp, len_sorted.tolist(),
                soft_samples=True, end_c = end_c, n_max=maxlen, auths=auths, adv_inp=True)
        rev_gen_samples = torch.cat([torch.unsqueeze(gs,0) for gs in rev_gen_samples],dim=0)

        # Undo the lensorting done on top
        gen_samples = gen_samples.index_select(1, rev_sort_idx)
        rev_gen_samples = rev_gen_samples.index_select(1, rev_sort_idx)
        rev_gen_lens = rev_gen_lens.index_select(0,rev_sort_idx.data)

        samples_out = (gen_samples, gen_lens, rev_gen_samples, rev_gen_lens)
    else:
        samples_out = (gen_samples, gen_lens)

    return (eval_out_gen_sort,) + eval_out_gt[0:1] + eval_out_gen[2:] + eval_out_gt[2:] + samples_out

def main(params):
    dp = DataProvider(params)

    # Create vocabulary and author index
    misc = {}
    if params['resume'] == None and params['loadgen']==None:
        if params['atoms'] == 'char':
            char_to_ix, ix_to_char = dp.createCharVocab(params['vocab_threshold'])
        else:
            char_to_ix, ix_to_char = dp.createWordVocab(params['vocab_threshold'])
        auth_to_ix, ix_to_auth= dp.createAuthorIdx()
        misc['char_to_ix'] = char_to_ix; misc['ix_to_char'] = ix_to_char; misc['auth_to_ix'] = auth_to_ix; misc['ix_to_auth'] = ix_to_auth
        restore_optim = False
        restore_gen = False
        restore_eval = False
    else:
        saved_model = torch.load(params['resume']) if params['loadgen'] == None else torch.load(params['loadgen'])
        model_gen_state = saved_model['state_dict_gen'] if params['loadgen'] == None else saved_model['state_dict']
        restore_gen = True
        if params['loadeval'] or params['resume']:
            saved_eval_model = torch.load(params['loadeval']) if params['loadeval'] else saved_model
            model_eval_state = saved_eval_model['state_dict_eval'] if params['loadeval'] == None else saved_eval_model['state_dict']
            eval_params = saved_eval_model['arch']
            restore_eval = True
            assert(not any([saved_eval_model['ix_to_char'][k] != saved_model['ix_to_char'][k] for k in saved_eval_model['ix_to_char']]))

        else:
            restore_eval = False
            eval_params = params
        if params['resume'] and not (params['loadgen'] or params['loadeval']):
            restore_optim = True
        else:
            restore_optim = False

        char_to_ix = saved_model['char_to_ix']
        auth_to_ix = saved_model['auth_to_ix']
        ix_to_char = saved_model['ix_to_char']
        misc['char_to_ix'] = char_to_ix; misc['ix_to_char'] = ix_to_char; misc['auth_to_ix'] = auth_to_ix
        if 'ix_to_auth' not in saved_model:
            misc['ix_to_auth'] = {auth_to_ix[a]:a for a in auth_to_ix}
        else:
            misc['ix_to_auth'] = saved_model['ix_to_auth']

    params['vocabulary_size'] = len(misc['char_to_ix'])
    params['num_output_layers'] = len(misc['auth_to_ix'])
    eval_params['generic_classifier'] = params['generic_classifier']

    modelGen = CharTranslator(params)
    modelEval = CharLstm(eval_params)
    # set to train mode, this activates dropout
    modelGen.train()
    modelEval.train()
    #Initialize the RMSprop optimizer

    optimGen = torch.optim.RMSprop(modelGen.parameters(),
            lr=params['learning_rate_gen'], alpha=params['decay_rate'],
            eps=params['smooth_eps'])
    optimEval = torch.optim.RMSprop(modelEval.parameters(),
            lr=params['learning_rate_eval'], alpha=params['decay_rate'],
            eps=params['smooth_eps'])

    mLcriterion = nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()
    # Do size averaging here so that classes are balanced
    bceLogitloss = nn.BCEWithLogitsLoss(size_average=True)
    eval_generic = nn.BCELoss(size_average=True)
    cycle_l1_loss = nn.L1Loss()
    accuracy_lay = nn.L1Loss()

    # Restore saved checkpoint
    if restore_gen:
        state = modelGen.state_dict()
        state.update(model_gen_state)
        modelGen.load_state_dict(state)
    if restore_eval:
        state = modelEval.state_dict()
        state.update(model_eval_state)
        modelEval.load_state_dict(state)
    if restore_optim:
        optimGen.load_state_dict(saved_model['gen_optimizer'])
        optimEval.load_state_dict(saved_model['eval_optimizer'])

    avg_acc_geneval = 0.
    avg_cyc_loss = 0.
    avg_err_eval = np.zeros(len(auth_to_ix))
    avgL_gen = 0.
    avgL_eval = 0. ;avgL_gt = 0.; avgL_fake = 0.; avgL_generic = 0.
    start_time = time.time()
    hiddenGen = modelGen.init_hidden(params['batch_size'])
    hid_zeros_gen = modelGen.init_hidden(params['batch_size'])
    hid_zeros_eval = modelEval.init_hidden(params['batch_size'])

    # Compute the iteration parameters
    epochs = params['max_epochs']
    total_seqs = dp.get_num_sents(split='train')
    iter_per_epoch = total_seqs // params['batch_size']
    total_iters = iter_per_epoch * epochs
    best_loss = 1000000.
    best_val = 1000.
    eval_every = int(iter_per_epoch * params['eval_interval'])

    skip_first = 40
    iters_eval= 1
    iters_gen = 1

    #val_score = eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])
    val_score = 0. #eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs = params['num_eval'])
    val_rank = 1000

    eval_function = eval_translator if params['mode'] == 'generative' else eval_classify
    leakage = 0. #params['leakage']

    disp_gen_samples(modelGen, dp, misc, maxlen=params['max_seq_len'], atoms=params['atoms'])
    ones = Variable(torch.ones(params['batch_size'])).cuda()
    zeros = Variable(torch.zeros(params['batch_size'])).cuda()
    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    print total_iters


    # log using crayon/tensorboard logger
    if params['tensorboard']:
        cc = CrayonClient(hostname=params['tensorboard'])
        # Create two experiments, one for generator and one for discriminator
        our_exp_name = "generator"+params['fappend']
        try:
            cc.remove_experiment(our_exp_name); cc.remove_experiment("discriminator"+params['fappend'])
        except ValueError:
            print 'No previous experiment of same name found'
        gen_log = cc.create_experiment(our_exp_name)
        disc_log = cc.create_experiment("discriminator"+params['fappend'])

    append_tensor = np.zeros((1,params['batch_size'],params['vocabulary_size']+ 1),dtype=np.float32)
    append_tensor[:,:,misc['char_to_ix']['2']] = 1
    append_tensor = Variable(torch.FloatTensor(append_tensor), requires_grad=False).cuda()


    for i in xrange(total_iters):
        # Update the evaluator and get it into a good state.
        it2=0
        #--------------------------------------------------------------------------
        # This is the loop to train evaluator
        #--------------------------------------------------------------------------
        while it2<(iters_eval*skip_first): #eval_acc <= 60. or gen_acc >= 45. or it2<iters_eval*skip_first:
            c_aid = np.random.choice(auth_to_ix.values())
            batch = dp.get_sentence_batch(params['batch_size'], split='train',
                            atoms=params['atoms'], aid= misc['ix_to_auth'][c_aid])
            inps, targs, auths, lens = dp.prepare_data(batch, misc['char_to_ix'],
                            misc['auth_to_ix'], maxlen=params['max_seq_len'])
            # outs are organized as
            outs =  adv_forward_pass(modelGen, modelEval, inps, targs, lens,
                        end_c=misc['char_to_ix']['.'], backprop_for='eval',
                        maxlen=params['max_seq_len'], auths=auths)
            targets = Variable(auths).cuda()

            #lossEval = lossGeneric
            optimEval.zero_grad()
            # Does this make any sense!!?
            # Split the classifiers. One for Obama and one for trump
            if 0:
                lossEvalGt = eval_criterion(outs[1], targets)
                lossEvalFake = eval_criterion(outs[0], targets)
                lossEval = lossEvalGt + lossEvalFake
            elif not params['wasserstien_loss']:
                #----------- ----Use two differnt losses for each classifier.-----------------------------#
                # 1. For GT samples- use the GT label and cross entropy loss to train both author classifiers.
                # 2. For generated samples - use the target author classifier and corresponding GT samples to classify
                # real vs fake using BCE loss.
                # Eg. we have : obama vs trump on real data + obama real vs obama fake (real data vs translated trump) +
                # trump real vs trump fake (real data vs translated obama)
                lossEvalGt = eval_criterion(outs[1], targets)
                total_loss_aid = 0.
                for aid in auth_to_ix.values():
                    real_aid_out = outs[1][:,aid].masked_select(targets==aid)
                    #TODO: This needs fix when there are more than one author
                    gen_aid_out = outs[0][:,aid].masked_select(targets== (1-aid))

                    if (targets==aid).data.all():
                        continue
                    elif not (targets==aid).data.any():
                        loss_aid = (bceLogitloss(gen_aid_out, zeros[:gen_aid_out.size(0)]))
                    else:
                        loss_aid = (bceLogitloss(real_aid_out, ones[:real_aid_out.size(0)]) +
                                   bceLogitloss(gen_aid_out, zeros[:gen_aid_out.size(0)]))
                    total_loss_aid = total_loss_aid + loss_aid
                    avg_err_eval[aid] += (gen_aid_out.data > 0.).float().mean()

                lossEval = total_loss_aid + lossEvalGt
                lossEval.backward()
            elif params['wasserstien_loss']:
                aid = c_aid
                real_aid_out = outs[1][:,aid].masked_select(targets==aid)
                #TODO: This needs fix when there are more than one author
                gen_aid_out = outs[0][:,aid].masked_select(targets== (1-aid))
                E_real, E_gen = real_aid_out.mean(), gen_aid_out.mean()
                loss_aid = E_gen - E_real

                if params['fisher_gan']:
                    var_real, var_fake = (real_aid_out**2).mean(), (gen_aid_out**2).mean()
                    constraint = (1 - (0.5*var_real + 0.5*var_fake))

                    loss_aid = loss_aid - alpha * constraint + params['weight_penalty']/2 * constraint**2
                    loss_aid.backward()
                else:
                    real_aid_out.backward(mone)
                    gen_aid_out.backward(one)

                avg_err_eval[aid] += (gen_aid_out.data > 0.).float().mean()
                lossEval = loss_aid# + lossEvalGt

                # This is used in improved wasserstien gan
                #if not params['fisher_gan']:
                #    _, fake_data = outs[-2].max(dim=-1)
                #    grad_penalty = calc_gradient_penalty(modelEval, targs.numpy(), np.array(lens), fake_data.data.cpu().numpy(),
                #            outs[-1].cpu().numpy())
                #else:
                #    variance_penalty

            #if params['generic_classifier']:
            #    lossGeneric = (eval_generic(outs[3], ones) + eval_generic(outs[2], zeros))
            #    #if i> 750:
            #    #    import ipdb; ipdb.set_trace()
            #    predGen_fake= outs[2]>0.5
            #    avg_acc_geneval += accuracy_lay(predGen_fake.float(),zeros).data[0]
            #    lossEval += lossGeneric
            #    avgL_generic += lossGeneric.data.cpu().numpy()[0]
            optimEval.step()
            avgL_gt += lossEvalGt.data.cpu().numpy()[0]
            avgL_fake += total_loss_aid.data.cpu().numpy()[0]
            avgL_eval += lossEval.data.cpu().numpy()[0]
            # Calculate discrim accuracy on generator samples.
            # This works only because it is binary target variable
            it2 += 1
            #print '%.2f'%lossEval.data[0],
        if it2>1:
            avg_err_eval = avg_err_eval/it2
            avgL_gt /= it2
            avgL_fake /= it2
            avgL_eval /= it2


        #===========================================================================

        #--------------------------------------------------------------------------
        # Training the Generator
        #--------------------------------------------------------------------------
        optimGen.zero_grad()
        c_aid = np.random.choice(auth_to_ix.values())
        batch = dp.get_sentence_batch(params['batch_size'], split='train', atoms=params['atoms'], aid =
                misc['ix_to_auth'][c_aid])
        inps, targs, auths, lens = dp.prepare_data(batch, misc['char_to_ix'], misc['auth_to_ix'],
                        maxlen=params['max_seq_len'])
        outs = adv_forward_pass(modelGen, modelEval, inps, targs, lens, end_c=misc['char_to_ix']['.'],
                maxlen=params['max_seq_len'], auths=auths, cycle_compute=True, append_symb=append_tensor)
        targets = Variable(auths).cuda()
        # Does this make any sense!!?
        cyc_targ = Variable(torch.cuda.FloatTensor(params['batch_size'],
                params['vocabulary_size']+1).zero_().scatter_add_(1, targs.transpose(0,1).cuda(),
                torch.ones(targs.size()[::-1]).cuda()).index_fill_(1,torch.cuda.LongTensor([0]),0), requires_grad=False)
        #cyc_targ.index_fill_(1,torch.cuda.LongTensor([0]),0)
        cyc_loss = 500. * cycle_l1_loss(outs[-2].sum(dim=0), cyc_targ)
        if 0:
            lossGen = eval_criterion(outs[0], ((-1*targets)+1)) + cyc_loss
        elif not params['wasserstien_loss']:
            total_loss_aid = 0.
            for aid in auth_to_ix.values():
                if (targets==aid).data.all():
                    continue
                gen_aid_out = outs[0][:,aid].masked_select(targets== (1-aid))
                loss_aid = bceLogitloss(gen_aid_out, ones[:gen_aid_out.size(0)])
                total_loss_aid = total_loss_aid + loss_aid
            lossGen = total_loss_aid
        elif params['wasserstien_loss']:
            aid = c_aid
            lossGen
            gen_aid_out = outs[0][:,aid].masked_select(targets== (1-aid))
            E_gen = gen_aid_out.mean()
            lossGen = -E_gen

        lossGen = lossGen + cyc_loss

        if params['generic_classifier']:
            lossGenericGenerator = eval_generic(outs[1], ones)
            lossGen += lossGenericGenerator
            #lossGen = 1*lossGenericGenerator
        #TODO
        lossGen.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(modelGen.parameters(), params['grad_clip'])
        # Take an optimization step
        optimGen.step()
        avgL_gen += lossGen.data.cpu().numpy()[0]
        avg_cyc_loss += cyc_loss.data.cpu().numpy()[0]
        #===========================================================================

        # Visualize some generator samples once in a while
        if i%100 ==99:
            disp_gen_samples(modelGen, dp, misc, maxlen=params['max_seq_len'], atoms=params['atoms'])
        skip_first = 1

        if i % params['log_interval'] == (params['log_interval']-1):
            interv = params['log_interval']
            lossGcyc = avg_cyc_loss / interv
            lossG = avgL_gen / interv
            lossEv_tot     = avgL_eval    / (interv*it2)
            lossEv_gt      = avgL_gt      / (interv*it2)
            lossEv_fake    = avgL_fake    / (interv*it2)
            lossEv_generic = avgL_generic / (interv*it2)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} | ms/it {:5.2f} | '
                            'loss - G {:3.2f} - Gc {:3.2f} - E {:3.2f} - Eg {:3.2f} - Ef {:3.2f} - EG {:3.2f}|'.format(
                i//iter_per_epoch, i, total_iters, params['learning_rate_gen'],
                elapsed * 1000 / args.log_interval, lossG, lossGcyc, lossEv_tot, lossEv_gt, lossEv_fake,
                lossEv_generic))

            if params['tensorboard']:
                gen_log.add_scalar_dict({'loss':lossG,'loss_cyc':lossGcyc}, step=i)
                disc_log.add_scalar_dict({'loss':lossEv_fake, 'loss_gt':lossEv_gt, 'err_auth0':avg_err_eval[0]/ interv,
                    'err_auth1':avg_err_eval[1]/ interv}, step=i)
                disc_log.add_scalar_value('err_generic', avg_acc_geneval/interv, step=i)
            avgL_gen = 0.
            avgL_eval = 0.
            avgL_gt= 0.
            avgL_fake = 0.
            avgL_generic = 0.
            avg_err_eval = np.zeros(len(auth_to_ix))
            avg_acc_geneval = 0.
            avg_cyc_loss = 0.

            if val_rank <=best_val and i>0:
                save_checkpoint({
                    'iter': i,
                    'arch': params,
                    'val_loss': val_rank,
                    'val_pplx': val_score,
                    'misc': misc,
                    'state_dict': modelGen.state_dict(),
                    'loss':  lossGen,
                    'optimizerGen' : optimGen.state_dict(),
                    'optimizerEval' : optimEval.state_dict(),
                }, fappend = params['fappend'],
                outdir = params['checkpoint_output_directory'])
                best_val = val_rank
            start_time = time.time()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', default='pan16AuthorMask', help='dataset: pan')
  # mode
  parser.add_argument('--mode', dest='mode', type=str, default='generative', help='print every x iters')
  parser.add_argument('--atoms', dest='atoms', type=str, default='char', help='character or word model')
  parser.add_argument('--maxpoolrnn', dest='maxpoolrnn', type=int, default=0, help='maximum sequence length')
  parser.add_argument('--pad_auth_vec', dest='pad_auth_vec', type=int, default=10, help='maximum sequence length')

  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=50, help='maximum sequence length')
  parser.add_argument('--vocab_threshold', dest='vocab_threshold', type=int, default=5, help='vocab threshold')

  parser.add_argument('--resume', dest='resume', type=str, default=None, help='append this string to checkpoint filenames')
  parser.add_argument('--loadgen', dest='loadgen', type=str, default=None, help='load generator parameters from this')
  parser.add_argument('--loadeval', dest='loadeval', type=str, default=None, help='load evaluator parameters from this')

  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=10, help='max batch size')
  parser.add_argument('--randomize_batches', dest='randomize_batches', type=int, default=1, help='randomize batches')

  # Optimization parameters
  parser.add_argument('--lr_gen', dest='learning_rate_gen', type=float, default=1e-4, help='solver learning rate')
  parser.add_argument('--lr_eval', dest='learning_rate_eval', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.95, help='solver learning rate')
  parser.add_argument('--lr_decay_st', dest='lr_decay_st', type=int, default=0, help='solver learning rate')

  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.99, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=10., help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--use_sgd', dest='use_sgd', type=int, default=0, help='Use sgd')
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=50, help='number of epochs to train for')

  parser.add_argument('--drop_prob_emb', dest='drop_prob_emb', type=float, default=0.25, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')

  # Validation args
  parser.add_argument('--eval_interval', dest='eval_interval', type=float, default=0.5, help='print every x iters')
  parser.add_argument('--num_eval', dest='num_eval', type=int, default=-1, help='print every x iters')
  parser.add_argument('--log', dest='log_interval', type=int, default=1, help='print every x iters')
  parser.add_argument('--tensorboard', dest='tensorboard', type=str, default=None, help='Send to specified tensorboard server')

  # LSTM parameters
  parser.add_argument('--en_residual_conn', dest='en_residual_conn', type=int, default=0, help='depth of hidden layer in generator RNNs')

  parser.add_argument('--embedding_size', dest='embedding_size', type=int, default=512, help='size of word encoding')
  # Generator's parameters
  parser.add_argument('--enc_hidden_depth', dest='enc_hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--enc_hidden_size', dest='enc_hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--dec_hidden_depth', dest='dec_hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--dec_hidden_size', dest='dec_hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')

  # Discriminator's parameters
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=1, help='depth of hidden layer in evaluator RNNs')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in eva;iatpr RNNs')
  parser.add_argument('--generic_classifier', dest='generic_classifier', default=False, action='store_true', help='Should we use a generic classifier to classify fake vs real text')

  parser.add_argument('--wass_loss', dest='wasserstien_loss', default=False, action='store_true', help='Use wassertien loss')
  parser.add_argument('--fisher_gan', dest='fisher_gan', default=False, action='store_true', help='Use fisher GAN penalty')
  parser.add_argument('--weight_penalty', dest='weight_penalty', type=float, default=1e-4)

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print json.dumps(params, indent = 2)
  main(params)
