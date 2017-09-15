import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from tqdm import tqdm
import tsne
import matplotlib
from matplotlib.pyplot import figure, show

data = json.load(open('../../../data/blogdata/dataset.json','r'))
infersent = torch.load('infersent.allnli.pickle')
infersent.set_glove_path('/BS/rshetty-wrk/work/code/author-mask/tools/InferSent/dataset/GloVe/glove.840B.300d.txt')
all_sents = []; all_age = []; all_gender = [];
sent_to_id = [];
all_sent_set = set()
for i,doc in tqdm(enumerate(data['docs'])):
    for j,st in enumerate(doc['tokens']):
        if st not in all_sent_set:
            all_sent_set.add(st)
            all_sents.append(' '.join(st.split()[1:-1]))
            sent_to_id.append([i,j])
            all_age.append(int(doc['attrib']['age']))
            all_gender.append(doc['attrib']['gender'])
all_age = np.array(all_age)
gender_map = {'male':0, 'female':1}
inv_gender_map = {0:'male', 1:'female'}
all_gender = np.array([gender_map[g] for g in all_gender])
sent_smallset = []
n_st_perclass = 500

for ag in np.unique(all_age):
    sent_smallset.append(np.random.choice(np.nonzero(all_age==ag)[0],n_st_perclass))

uniqAge = np.unique(all_age)
sent_smallset_idx = np.concatenate(sent_smallset)
sent_subset = [all_sents[i] for i in sent_smallset_idx]
infersent.build_vocab(sent_subset,tokenize=False)
all_encoding = infersent.encode(sent_subset,bsize=128,tokenize=False,verbose=True)


tsnEmbSubset = tsne.bh_sne(all_encoding.astype('float64'),pca_d=512)

cmap = matplotlib.cm.get_cmap('Spectral')
def onpick3(event):
    ind = event.ind
    print ''
    for idx in ind:
        print 'onpick3 scatter: %5d - Age %2d, Gender %6s:  %s'%( idx, uniqAge[(idx//n_st_perclass)], inv_gender_map[all_gender[sent_smallset_idx[idx]]],  sent_subset[idx])

fig = figure()
ax1 = fig.add_subplot(111)
legendList = [str(ag) for ag in uniqAge]
sc = ax1.scatter(tsnEmbSubset[:,0],tsnEmbSubset[:,1],c=all_age[sent_smallset_idx], picker=True, cmap=cmap,alpha=0.6)
fig.canvas.mpl_connect('pick_event', onpick3)
plt.title('T-Sne embedding of semantic sentence encoding for different age groups')
plt.colorbar(sc)

fig2 = figure()
ax2 = fig2.add_subplot(111)
sc = ax2.scatter(tsnEmbSubset[:,0],tsnEmbSubset[:,1],c=all_gender[sent_smallset_idx], picker=True, cmap=cmap,alpha=0.6)
fig2.canvas.mpl_connect('pick_event', onpick3)
plt.title('T-Sne embedding of semantic sentence encoding - Male (0) Vs Female (1)')
plt.colorbar(sc)

plt.show()
