import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

def get_classifier(params):
    from char_lstm import CharLstm
    from char_cnn import CharCNN

    if params.get('modeltype', 'lstm') == 'lstm':
        return CharLstm(params)
    else:
        return CharCNN(params)

def packed_mean(packed, dim=0):
    unp = pad_packed_sequence(packed)
    unp_data = unp[0]
    lens = unp[1]
    sizes = unp_data.size()
    mean = torch.sum(unp_data,dim=dim, keepdim=False)/Variable(torch.FloatTensor(lens).view(-1,1).expand(sizes[1],sizes[2]),requires_grad=False).cuda()
    return mean

def packed_add(p1,p2):
    # Assuming sequence lengths are fixed
    unp1 = pad_packed_sequence(p1); unp2 = pad_packed_sequence(p2)
    unp1_data = unp1[0] ;  unp2_data = unp2[0] ;
    lens = unp1[1]
    sum1_2 = pack_padded_sequence(unp1_data + unp2_data, lens)
    return sum1_2
