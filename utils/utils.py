from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
import sklearn
from sklearn.metrics import roc_auc_score
import cPickle as pickle
from collections import defaultdict
import torch.nn.functional as FN

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    elif h == None:
        return None
    else:
        return tuple(repackage_hidden(v) for v in h)

def eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs=-1,
        dump_scores=False):
    #We go one document at a time and evaluate it using all the authors
    hidden = model.init_hidden(1)
    correct = 0.
    n_docs = 0.
    mean_corr_prob = 0.
    mean_max_prob = 0.
    mean_min_prob = 0.
    mean_rank = 0.
    current_doc_score = np.zeros(model.num_output_layers)
    print '--------------Runnin Eval now ---------------------'
    all_doc_scores = []
    all_window_scores = defaultdict(list)
    all_auths = []
    for i, b_data in tqdm(enumerate(dp.iter_single_doc(split=split, max_docs=max_docs))):
        done = b_data[1]
        inps, targs, auths, lens = dp.prepare_data(b_data[0], char_to_ix, auth_to_ix)
        output, hidden = model.forward_eval(inps, hidden, compute_softmax=True)
        z = output.data.cpu().numpy()
        scores = z[np.arange(lens[0]),:,np.squeeze(targs.numpy()[:lens[0]])].sum(axis=0)
        # Accumulate the scores for each doc.
        current_doc_score = current_doc_score + scores
        if dump_scores:
            all_window_scores[n_docs].append(scores)

        if done:
            hidden[0].data.index_fill_(1,torch.LongTensor([0]).cuda(),0.)
            hidden[1].data.index_fill_(1,torch.LongTensor([0]).cuda(),0.)
            correct = correct + (current_doc_score.argmax() == auths[0])
            mean_rank = mean_rank + np.where(current_doc_score.argsort()[::-1]==auths[0])[0][0]
            mean_corr_prob = mean_corr_prob + current_doc_score[auths[0]]
            mean_max_prob = mean_max_prob + current_doc_score.max()
            mean_min_prob = mean_min_prob + current_doc_score.min()
            n_docs = n_docs + 1.
            all_doc_scores.append(current_doc_score[None,:])
            all_auths.append(auths.numpy())
            # Reset the doc probs
            current_doc_score = np.zeros(model.num_output_layers)

    if dump_scores:
        pickle.dump({'scores':all_window_scores,'authors':all_auths}, open('window_scores_'+split+'.p','w'))

    print 'Eval on %.1f docs of %s set is done'%(n_docs, split)
    print 'Accuracy is %.3f., mean rank is %.2f '%(100. * (correct/n_docs), mean_rank/n_docs)
    print 'Corr is %.2f | Max is %.2f | Min is %.2f'%(mean_corr_prob/n_docs, mean_max_prob/n_docs, mean_min_prob/n_docs)
    print '-----------------------------------'

    all_auths = np.concatenate(all_auths, axis=0)
    all_docs = np.concatenate(all_doc_scores, axis=0)
    all_docs = all_docs - all_docs.mean(axis=1)[:,None]

    # Now we can look at correct author - doc pairs
    # But we can also augment with incorrect author-doc pairs
    # So pick one random incorrect author for each doc and see how the doc ranks as per that author
    neg_auths = np.random.randint(0,model.num_output_layers, int(n_docs))

    adjusted_scores = ((np.argsort(all_docs[:, np.concatenate([all_auths, neg_auths])],axis=0)==np.concatenate([np.arange(n_docs), np.arange(n_docs)])).argmax(axis=0)+1)/n_docs
    print 'Accuracy per adjusted scores %.3f'%(100.*((adjusted_scores[:int(n_docs)] >= 0.5).sum()+(adjusted_scores[int(n_docs):] < 0.5).sum())/(2.*n_docs))
    auc = roc_auc_score(np.concatenate([np.ones(int(n_docs),dtype=int), np.zeros(int(n_docs),dtype=int)]), adjusted_scores)
    print 'AUC is  %.2f'%(auc)

    return (mean_rank/n_docs), auc

def initdumpdict():
    return {'scores':[]}

def eval_classify(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs=-1,
        dump_scores=False, dump_label = None):
    #We go one document at a time and evaluate it using all the authors
    b_sz = 1000
    hidden_zero = model.init_hidden(b_sz)
    c_sz = b_sz
    correct = 0.
    correct_topk = 0.
    n_docs = 0.
    mean_corr_prob = 0.
    mean_max_prob = 0.
    mean_min_prob = 0.
    mean_rank = 0.
    n_auths = len(auth_to_ix)
    current_doc_score = np.zeros(n_auths)
    print '--------------Runnin Eval now ---------------------'
    all_doc_scores = []
    all_window_scores = defaultdict(initdumpdict)
    all_auths = []
    correct_textblock = 0.
    correct_textblock_perclass = np.zeros(n_auths)
    ix_to_auth = {auth_to_ix[a]: a for a in auth_to_ix}
    block_perclass = np.zeros(n_auths)
    correct_textblock_topk = 0.
    n_blks = 0.
    doc_count = np.zeros(n_auths)
    doc_correct = np.zeros(n_auths)
    for i, b_data in tqdm(enumerate(dp.iter_sentences(split=split, atoms=params.get('atoms','char'), batch_size = b_sz))):
    #for i, b_data in tqdm(enumerate(dp.iter_single_doc(split=split, max_docs=max_docs))):
        if len(b_data[0]) != b_sz or len(b_data[0]) != c_sz:
            hidden_zero =  model.init_hidden(len(b_data[0]))
            c_sz = len(b_data[0])
        done = b_data[1]
        inps, targs, auths, lens = dp.prepare_data(b_data[0], char_to_ix, auth_to_ix)
        output, _ = model.forward_classify(inps, hidden_zero, compute_softmax=False,
                predict_mode=True, lens=lens)
        output = FN.log_softmax(output.contiguous(), dim=-1)
        z = output.data.cpu().numpy()
        scores = z
        correct_textblock = correct_textblock + (scores.argmax(axis=1) == auths.numpy()).sum()
        correct_textblock_topk += (np.where(scores.argsort(axis=1)[:,::-1]==auths.numpy()[:,None])[1]<=params.get('topk',5)).sum()
        n_blks = n_blks+c_sz

        # Accumulate accuracies per class
        np.add.at(correct_textblock_perclass, auths.numpy(), (scores.argmax(axis=1) == auths.numpy()))
        np.add.at(block_perclass, auths.numpy(), 1.)

        # Accumulate the scores for each doc.
        current_doc_score = current_doc_score + scores.sum(axis=0)
        if dump_scores:
            all_window_scores[n_docs]['scores'].append({'score':scores,'sid':[btch['sid'] for btch in b_data[0]]})

        if done:
            correct = correct + (current_doc_score.argmax() == auths[0])
            doc_correct[auths[0]] += (current_doc_score.argmax() == auths[0])
            doc_count[auths[0]] += 1.
            mean_rank = mean_rank + np.where(current_doc_score.argsort()[::-1]==auths[0])[0][0]
            correct_topk += (np.where(current_doc_score.argsort()[::-1]==auths[0])[0][0]<=params.get('topk',5)).sum()
            mean_corr_prob = mean_corr_prob + current_doc_score[auths[0]]
            mean_max_prob = mean_max_prob + current_doc_score.max()
            mean_min_prob = mean_min_prob + current_doc_score.min()
            all_doc_scores.append(current_doc_score[None,:])
            all_auths.append(auths.numpy())
            # Reset the doc probs
            current_doc_score = np.zeros(n_auths)
            if dump_scores:
                all_window_scores[n_docs]['attrib'] = b_data[0][0]['attrib']
                all_window_scores[n_docs]['id'] = b_data[0][0]['id']
            n_docs = n_docs + 1.

    if dump_scores:
        pickle.dump({'scores':all_window_scores,'authors':all_auths}, open('window_scores_'+'blog-age_teenVsRes_dep2res_0.5drop_uniformlen_0.68'+split+'.p','w'))

    topk = 5
    doc_acc = (100. * (correct/n_docs))
    print 'Eval on %.1f docs of %s set is done'%(n_docs, split)
    print 'Doc level accuracy is %.3f., mean rank is %.2f '%(100. * (correct/n_docs), mean_rank/n_docs)
    fp = doc_count[1]- doc_correct[1]
    recall = doc_correct[0]/doc_count[0]
    precision = doc_correct[0]/(doc_correct[0]+fp)
    f1score = 2.*(precision*recall)/(precision+recall)
    print 'Precision is %.2f : Recall is %.2f , F1-score is %.2f'%(precision, recall, f1score)

    print 'Top-%d Accuracy is %.2f'%(params.get('topk',5), 100.*(correct_topk/n_docs))
    print ''
    print 'Block level accuracy is %.3f.'%(100. * (correct_textblock/n_blks))
    print 'Top-%d Block Accuracy is %.2f'%(params.get('topk',5), 100. *(correct_textblock_topk/n_blks))

    for i in xrange(n_auths):
        print 'Block level accuracy of class %s is %.2f.'%(ix_to_auth[i], 100. * (correct_textblock_perclass[i]/block_perclass[i]))
    fp = block_perclass[1]-correct_textblock_perclass[1]
    recall = correct_textblock_perclass[0]/block_perclass[0]
    precision = correct_textblock_perclass[0]/(correct_textblock_perclass[0]+fp)
    f1score = 2.*(precision*recall)/(precision+recall)
    print 'Precision is %.2f : Recall is %.2f , F1-score is %.2f'%(precision, recall, f1score)


    print ''
    print 'Corr is %.2f | Max is %.2f | Min is %.2f'%(mean_corr_prob/n_docs, mean_max_prob/n_docs, mean_min_prob/n_docs)
    print '-----------------------------------'

    all_auths = np.concatenate(all_auths, axis=0)
    all_docs = np.concatenate(all_doc_scores, axis=0)
    all_docs = all_docs - all_docs.mean(axis=1)[:,None]

    # Now we can look at correct author - doc pairs
    # But we can also augment with incorrect author-doc pairs
    # So pick one random incorrect author for each doc and see how the doc ranks as per that author
    if 0 :#model.num_output_layers >2:
        neg_auths = np.random.randint(0,n_auths, int(n_docs))
        adjusted_scores = ((np.argsort(all_docs[:, np.concatenate([all_auths, neg_auths])],axis=0)==np.concatenate([np.arange(n_docs), np.arange(n_docs)])).argmax(axis=0)+1)/n_docs
        print 'Accuracy per adjusted scores %.3f'%(100.*((adjusted_scores[:int(n_docs)] >= 0.5).sum()+(adjusted_scores[int(n_docs):] < 0.5).sum())/(2.*n_docs))
        auc = roc_auc_score(np.concatenate([np.ones(int(n_docs),dtype=int), np.zeros(int(n_docs),dtype=int)]), adjusted_scores)
        print 'AUC is  %.2f'%(auc)
    else:
        auc = 0.

    return doc_acc,(correct_textblock/n_blks)

def eval_translator(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs=-1,
        dump_scores=False):
    #We go one document at a time and evaluate it using all the authors
    b_sz = 100
    c_sz = b_sz
    hidden_zero = model.init_hidden(b_sz)
    correct = 0.
    n_docs = 0.
    total_loss = 0.
    #current_doc_score = np.zeros(model.num_output_layers)
    print '--------------Runnin Eval now ---------------------'
    criterion = nn.CrossEntropyLoss()

    # Put the model in testing mode
    model.eval()

    for i, b_data in tqdm(enumerate(dp.iter_sentences(split=split, atoms=params.get('atoms','char'), batch_size = b_sz))):
        if len(b_data[0]) != b_sz or len(b_data[0]) != c_sz:
            hidden_zero =  model.init_hidden(len(b_data[0]))
            c_sz = len(b_data[0])
        inps, targs, auths, lens = dp.prepare_data(b_data[0], char_to_ix, auth_to_ix, maxlen=params['max_seq_len'])
        output, hidden = model.forward_mltrain(inps, lens, inps, lens, hidden_zero, compute_softmax=False, auths=auths)
        targets = pack_padded_sequence(Variable(targs).cuda(),lens)
        loss = criterion(pack_padded_sequence(output,lens)[0], targets[0])
        total_loss += loss.data.cpu().numpy()[0]

    cur_loss = total_loss / i
    perplexity = np.exp(cur_loss)

    print 'Eval on %.1f sentences of %s set is done'%(i*b_sz, split)
    print 'Perplexity is %.2f '%(perplexity)
    print '-----------------------------------'

    model.train()

    return cur_loss, perplexity
