import torch
import torch.nn.init as init


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data)
        init.xavier_normal(m.bias.data)
    if classname.find('Linear') != -1:
        init.xavier_uniform(m.weight.data)
    if classname.find('LSTM') != -1:
        init.xavier_uniform(m.weight_hh_l0)
        init.xavier_uniform(m.weight_ih_l0)

def kaiming_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data)
        init.kaiming_normal(m.bias.data)
    if classname.find('Linear') != -1:
        init.kaiming_uniform(m.weight.data)

def orthogonal_init(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal(m.weight_hh_l0)
        init.orthogonal(m.weight_ih_l0)