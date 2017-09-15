import argparse
import json
from models.char_translator import CharTranslator
from collections import defaultdict
from utils.data_provider import DataProvider
from torch.autograd import Variable, Function
import torch.nn.functional as FN
from tqdm import tqdm

from models.fb_semantic_encoder import BLSTMEncoder
import torch



def main(params):
    saved_model = torch.load(params['checkpoint'])
    cp_params = saved_model['arch']
    dp = DataProvider(cp_params)

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

    def process_batch(batch, featstr = 'sent_enc'):
        inps, _, _,lens = dp.prepare_data(batch, char_to_ix, auth_to_ix, maxlen=cp_params['max_seq_len'])
        enc_out = modelGenEncoder.forward_encode(inps, lens)
        enc_out = enc_out.data.cpu().numpy()
        for i,b in enumerate(batch):
            res['docs'][b['id']]['sents'][b['sid']][featstr] = enc_out[i,:].tolist()

    if params['use_semantic_encoder']:
        modelGenEncoder = BLSTMEncoder(char_to_ix, ix_to_char, params['glove_path'])
        encoderState = torch.load(params['use_semantic_encoder'])
    else:
        modelGenEncoder = CharTranslator(cp_params, encoder_only=True)
        encoderState = model_gen_state

    state = modelGenEncoder.state_dict()
    for k in encoderState:
        if k in state:
            state[k] = encoderState[k]
    modelGenEncoder.load_state_dict(state)
    modelGenEncoder.eval()

    resf = params['resfile']
    res = json.load(open(resf,'r'))
    bsz = params['batch_size']

    batch = []
    print ' Processing original text'
    for i in tqdm(xrange(len(res['docs']))):
        ix = auth_to_ix[res['docs'][i]['author']]
        for j in xrange(len(res['docs'][i]['sents'])):
            st = res['docs'][i]['sents'][j]['sent'].split()
            if len(st) > 0:
                batch.append({'in': st,'targ': st, 'author': res['docs'][i]['author'],
                    'id':i, 'sid': j})
            if len(batch) == bsz:
                process_batch(batch, featstr = 'sent_enc')
                batch = []
    if batch:
        process_batch(batch, featstr = 'sent_enc')
        batch = []

    print 'Processing translated text'
    for i in tqdm(xrange(len(res['docs']))):
        ix = auth_to_ix[res['docs'][i]['author']]
        for j in xrange(len(res['docs'][i]['sents'])):
            st = res['docs'][i]['sents'][j]['trans_sent'].split()
            if len(st) > 0:
                batch.append({'in': st,'targ': st, 'author': res['docs'][i]['author'],
                    'id':i, 'sid': j})
            if len(batch) == bsz:
                process_batch(batch, featstr = 'trans_enc')
                batch = []
    if batch:
        process_batch(batch, featstr = 'trans_enc')
        batch = []

    json.dump(res, open(resf+'test','w'))


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-r','--resfile', dest='resfile', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-n','--ndisp', dest='ndisp', type=int, default=10, help='batch_size to use')
  parser.add_argument('-a','--age', dest='age', type=int, default=1, help='batch_size to use')
  parser.add_argument('-o','--offset', dest='offset', type=int, default=0, help='batch_size to use')
  parser.add_argument('-c','--checkpoint', dest='checkpoint', type=str, default=None, help='generator/GAN checkpoint filename')
  parser.add_argument('-b','--batch_size', dest='batch_size', type=int, default=100, help='generator/GAN checkpoint filename')
  parser.add_argument('-g','--glove_path', dest='glove_path', type=str, default='data/glove.840B.300d.txt', help='generator/GAN checkpoint filename')
  parser.add_argument('--use_semantic_encoder', dest='use_semantic_encoder', type=str, default=None, help='generator/GAN checkpoint filename')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
