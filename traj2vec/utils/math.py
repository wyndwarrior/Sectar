import numpy as np
import torch

c = float(0.5 * np.log(2*np.pi))
#c = torch.cuda.FloatTensor([float(0.5 * np.log(2*np.pi).astype(np.float32))])

def product(list):
    value = list[0]
    for v in list[1:]:
        value *= v
    return value

def normal_log_pdf(mean, logstd, x, eps=0.0):
    bs, seq_len, k = x.size()
    zs = (x - mean) / torch.exp(logstd)
    return - torch.sum(logstd, -1) - 0.5 * torch.sum(zs.pow(2), -1) - 0.5 * k * np.log(2 * 3.141592)
    #return k*c - torch.sum(logvar / 2 - (x - mean).pow(2) / (2 * (torch.exp(logvar))), 2).squeeze()
