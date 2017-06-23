import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


def packed_mean(packed, dim=0):
    unp = pad_packed_sequence(packed)
    unp_data = unp[0]
    lens = unp[1]
    sizes = unp_data.size()
    mean = torch.sum(unp_data,dim=dim, keepdim=False)/Variable(torch.FloatTensor(lens).view(-1,1).expand(sizes[1],sizes[2]),requires_grad=False).cuda()
    return mean
