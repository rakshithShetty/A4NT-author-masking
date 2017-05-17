from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
import sklearn
from sklearn.metrics import roc_auc_score


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def eval_model(dp, model, params, char_to_ix, auth_to_ix, split='val', max_docs=-1):
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
    all_auths = []
    for i, b_data in tqdm(enumerate(dp.iter_single_doc(split=split, max_docs=max_docs))):
        done = b_data[1]
        inps, targs, auths, lens = dp.prepare_data(b_data[0], char_to_ix, auth_to_ix)
        output, hidden = model.forward_eval(inps, hidden)
        z = output.data.cpu().numpy()
        scores = z[np.arange(lens[0]),:,np.squeeze(targs.numpy()[:lens[0]])].sum(axis=0)
        # Accumulate the scores for each doc.
        current_doc_score = current_doc_score + scores

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


