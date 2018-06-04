import argparse
import json
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from utils.data_provider import DataProvider
import string
from collections import defaultdict
from collections import Counter
import re
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from models.mlp_classifier import MLP_classifier

def normalize(X, meanX=None, stdX = None):
    # X is assumed to be n_data x n_feats
    if meanX is None:
        meanX = X.mean(axis=0)
        stdX = X.std(axis=0)
    norm_X = (X-meanX)/(stdX+1e-10)
    return norm_X, meanX, stdX

def count(dp, vocab, auth_to_ix, split='train'):
    # Now build feature vectors for all docs:
    docCounts= np.zeros((len(dp.splits[split]), len(vocab)),dtype=np.float32)
    target = np.zeros((len(dp.splits[split])),dtype=np.int)

    for i,idx in enumerate(dp.splits[split]):
        for w in dp.data['docs'][idx]['tokenized']:
            if w in vocab:
                docCounts[i,vocab[w]] = docCounts[i,vocab[w]] + 1.
        target[i] = auth_to_ix[dp.data['docs'][idx]['author']]

    return docCounts, target

def bow_features(counts, tfidf=False, idf=None):
    if tfidf:
        # Compute TF and IDF
        tf = counts / (counts.sum(axis=1)[:,None]+1e-8)
        if idf is None:
            idf = np.log(float(counts.shape[0])/(counts>0).sum(axis=0))
        bow_features = tf*idf[None,:]
    else:
        # Just compute TF
        bow_features = counts / (counts.sum(axis=1)[:,None] + 1e-8)

    return bow_features, idf

def transform_labels(inplabels, targ_class):
    targIds = (inplabels==targ_class)
    labels = inplabels.copy()
    labels[targIds] = 1
    labels[~targIds] = -1
    return labels

def main(params):
    dp = DataProvider(params)
    auth_to_ix = dp.createAuthorIdx()

    # Preprocess the training data
    train_docs = []
    targets = []
    model = {}

    # remove numbers
    bad_hombres = range(10)
    if params['nostop']:
        bad_hombres = bad_hombres + stopwords.words('english')
    if params['nopunct']:
        bad_hombres = bad_hombres + list(string.punctuation)

    bad_hombres = set(bad_hombres)

    all_words = Counter()

    for i,doc in enumerate(dp.data['docs']):
        no_num = re.sub(r'\d+', '', doc['text'].lower())
        curr_text = [w for w in wordpunct_tokenize(no_num) if w not in bad_hombres]
        dp.data['docs'][i]['tokenized'] = curr_text
        if doc['split'] == 'train':
            all_words.update(curr_text)

    short_vocab = {w:i for i,w in enumerate([wrd for wrd in all_words if all_words[wrd] > params['vocab_threshold']])}

    docCounts_train, target_train = count(dp, short_vocab, auth_to_ix, split='train')
    bow_features_train, idf_train = bow_features(docCounts_train, params['tfidf'])

    docCounts_val , target_val = count(dp, short_vocab, auth_to_ix, split='val')
    bow_features_val, _ = bow_features(docCounts_val, params['tfidf'], idf=idf_train)

    # Do PCA?
    if params['pca'] >0:
        pca_model = PCA(n_components=params['pca'])
        bow_features_train = pca_model.fit_transform(bow_features_train)
        print'Explained variance is %.2f'%(sum(pca_model.explained_variance_ratio_))

        bow_features_val = pca_model.transform(bow_features_val)
        params['pca'] = bow_features_train.shape[-1]

    # Normalize the data
    bow_features_train, mean_tr, std_tr = normalize(bow_features_train)
    bow_features_val , _, _ = normalize(bow_features_val , mean_tr, std_tr)


    if params['mlp'] == False:
        if params['linearsvm']:
            # Linear SVC alread implements one-vs-rest
            svm_model = LinearSVC()#verbose=1)
            svm_model.fit(bow_features_train, target_train)

        #Time to evaluate now.
        confTr = svm_model.decision_function(bow_features_train)
        confVal = svm_model.decision_function(bow_features_val)
    else:
        params['num_output_layers'] =len(auth_to_ix)
        params['inp_size'] = params['pca']
        model = MLP_classifier(params)
        model.fit(bow_features_train, target_train, bow_features_val, target_val, params['epochs'], params['lr'], params['l2'])
        confTr = model.decision_function(bow_features_train)
        confVal = model.decision_function(bow_features_val)

    mean_rank_train = np.where(confTr.argsort(axis=1)[:,::-1] == target_train[:,None])[1].mean()
    topk_train = (np.where(confTr.argsort(axis=1)[:,::-1] == target_train[:,None])[1] <= params['topk']).sum() * 100. / len(target_train)
    train_accuracy = 100. * float((confTr.argmax(axis=1) == target_train).sum()) / len(target_train)

    mean_rank_val = np.where(confVal.argsort(axis=1)[:,::-1] == target_val[:,None])[1].mean()
    topk_val = (np.where(confVal.argsort(axis=1)[:,::-1] == target_val[:,None])[1] <= params['topk']).sum() * 100. / len(target_val)
    val_accuracy = 100. * float((confVal.argmax(axis=1) == target_val).sum()) / len(target_val)

    # DO the binary evaluation similar to the Bagnall
    #confTr = confTr - confTr.mean(axis=1)[:,None]
    n_auths = len(auth_to_ix)

    n_train = confTr.shape[0]
    neg_auths_tr = np.random.randint(0,n_auths,n_train)
    adjusted_scores_tr = ((np.argsort(confTr[:, np.concatenate([target_train.astype(int), neg_auths_tr])],axis=0)==np.concatenate([np.arange(n_train), np.arange(n_train)])).argmax(axis=0)+1)/float(n_train)
    auc_tr = roc_auc_score(np.concatenate([np.ones(int(n_train),dtype=int), np.zeros(int(n_train),dtype=int)]), adjusted_scores_tr)

    n_val = confVal.shape[0]
    neg_auths_val = np.random.randint(0,n_auths,n_val)
    adjusted_scores_val = ((np.argsort(confVal[:, np.concatenate([target_val.astype(int), neg_auths_val])],axis=0)==np.concatenate([np.arange(n_val), np.arange(n_val)])).argmax(axis=0)+1)/float(n_val)
    auc_val = roc_auc_score(np.concatenate([np.ones(int(n_val),dtype=int), np.zeros(int(n_val),dtype=int)]), adjusted_scores_val)

    print '------------- Training set-------------------'
    print 'Accuracy is %.2f, Mean rank is %.2f / %d'%(train_accuracy, mean_rank_train, len(auth_to_ix))
    print 'Top-%d Accuracy is %.2f'%(params['topk'], topk_train)
    print 'Accuracy per adjusted scores %.3f'%(100.*((adjusted_scores_tr[:n_train] >= 0.5).sum()+(adjusted_scores_tr[n_train:] < 0.5).sum())/(2.*n_train))
    print 'AUC is  %.2f'%(auc_tr)

    print '------------- Val set-------------------'
    print 'Accuracy is %.2f, Mean rank is %.2f / %d'%(val_accuracy, mean_rank_val, len(auth_to_ix))
    print 'Top-%d Accuracy is %.2f'%(params['topk'], topk_val)
    print 'Accuracy per adjusted scores %.3f'%(100.*((adjusted_scores_val[:n_val] >= 0.5).sum()+(adjusted_scores_val[n_val:] < 0.5).sum())/(2.*n_val))
    print 'AUC is  %.2f'%(auc_val)

    print '--------------------------------------------------------------------------'
    print '--------------------------------------------------------------------------\n\n'



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', default='pan16AuthorMask', help='dataset: pan')

  parser.add_argument( '--nopunct', dest='nopunct', default=False, action='store_true', help='dataset: pan')
  parser.add_argument( '--nostop', dest='nostop', default=False, action='store_true', help='dataset: pan')
  parser.add_argument( '--tfidf', dest='tfidf', default=False, action='store_true', help='dataset: pan')

  parser.add_argument( '--pca', dest='pca', type=int, default=-1, help='dataset: pan')

  parser.add_argument( '--linearsvm', dest='linearsvm', default=False, action='store_true', help='dataset: pan')

  parser.add_argument( '--mlp', dest='mlp', default=False, action='store_true', help='use mlp as the learning model')
  parser.add_argument( '--hidden_widths', dest='hidden_widths', nargs='+', type=int, default=[], help='hidden layer configuration')
  parser.add_argument( '--lr', dest='lr', type=float, default=1e-3, help='learning rate')
  parser.add_argument( '--l2', dest='l2', type=float, default=1e-2, help='learning rate')
  parser.add_argument( '--epochs', dest='epochs', type=int, default=200, help='learning rate')
  parser.add_argument( '--topk', dest='topk', type=int, default=5, help='learning rate')

  # Vocab threshold
  parser.add_argument('--vocab_threshold', dest='vocab_threshold', type=int, default=5, help='vocab threshold')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print json.dumps(params, indent = 2)
  main(params)


