import torch
import torch.nn as nn
from traj2vec.nn.weight_inits import xavier_init, orthogonal_init
from traj2vec.utils.torch_utils import Variable

class RNN(nn.Module):
    def __init__(self, rnn_network, hidden_dim):
        super(RNN, self).__init__()
        self.rnn_network = rnn_network
        self.h_size = 1
        if self.rnn_network.bidirectional:
            self.h_size += 1
        self.h_size *= self.rnn_network.num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.apply(xavier_init)
        self.rnn_network.apply(orthogonal_init)

    def init_hidden(self, bs):
        self.hidden = (Variable(torch.zeros(self.h_size, bs, self.hidden_dim)),
                       Variable(torch.zeros(self.h_size, bs, self.hidden_dim)))

    def set_state(self, state):
        self.hidden = torch.split(state, self.h_size, 0)

    def get_state(self):
        return torch.cat(self.hidden, 0)

    def forward(self, x):
        # x is (seq_len, bs, input_dim)
        out, self.hidden = self.rnn_network(x, self.hidden)
        return out


