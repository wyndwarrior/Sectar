import numpy as np
import torch.nn as nn
from traj2vec.utils.torch_utils import from_numpy
import torch

class Parameter(nn.Module):
    def __init__(self, input_dim, output_dim, init):
        super(Parameter, self).__init__()
        self.output_dim = output_dim
        self.init = init
        self.param_init = from_numpy(np.zeros((1, output_dim)) + init).float()
        #TODO: fix this nn.Parameter(self.param_init) 
        self.params_var = nn.Parameter(self.param_init) #torch.autograd.Variable(self.param_init, requires_grad=True)

    def parameters(self):
        return [self.params_var]

    def forward(self, x):
        batch_size = x.size()[0]
        return self.params_var.repeat(batch_size, 1) #self.output_dim)


    def reset_weights(self):
        self.params_var.data.fill_(self.init)