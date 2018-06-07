import torch
from traj2vec.math.distributions.categorical import Categorical, RecurrentMultiCategorical, RecurrentCategorical
from traj2vec.models.containers.module_container import ModuleContainer
from traj2vec.nn.weight_inits import xavier_init
from traj2vec.utils.torch_utils import Variable, get_numpy, np_to_var
import numpy as np
import torch.nn as nn
from traj2vec.nn.rnn import RNN

class CategoricalNetwork(ModuleContainer):
    def __init__(self, prob_network, output_dim, obs_filter=None):
        super().__init__()
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.modules = [self.prob_network]
        self.obs_filter = obs_filter

    def forward(self, x):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        prob = self.prob_network(x)
        dist = Categorical(prob)
        return dist

    def reset(self, bs):
        pass


class LSTMPolicy(ModuleContainer):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.rnn = RNN(self.lstm, hidden_dim)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.modules = [self.rnn, self.linear]

    def reset(self, bs):
        self.rnn.init_hidden(bs)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        lstm_out = self.rnn.forward(x)
        lstm_reshape = lstm_out.view((-1, self.hidden_dim))
        output = self.softmax(self.linear(lstm_reshape))
        dist = Categorical(output)
        return dist

    def set_state(self, state):
        self.rnn.set_state(state)

    def get_state(self):
        return self.rnn.get_state()
    def recurrent(self):
        return True

class RecurrentCategoricalPolicy(CategoricalNetwork):
    def __init__(self, recurrent_network, prob_network, path_len, output_dim):
        # Base network is shared
        # Prob networks is list of head networks that'll each output a categorical
        super().__init__(prob_network, output_dim)
        self.recurrent_network = recurrent_network
        self.path_len=path_len
        self.modules = [self.recurrent_network, self.prob_network]
        self.apply(xavier_init)

    def init_input(self, bs):
        # Return one hot for each cat
        return Variable(torch.zeros(bs, self.output_dim))

    def set_state(self, state):
        self.recurrent_network.set_state(state)

    def get_state(self):
        return self.recurrent_network.get_state()

    def forward(self, x):
        # Torch uses (path_len, bs, input_dim) for recurrent input
        # path_len must be 1
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        assert x.size()[0] == 1, "Path len must be 1"
        out = self.recurrent_network(x).squeeze(0)
        prob = self.prob_network(out)
        return Categorical(prob)

    def reset(self, bs):
        self.recurrent_network.init_hidden(bs)

    def recurrent(self):
        return True

class RecurrentCategoricalNetwork(CategoricalNetwork):
    def __init__(self, recurrent_network, prob_network, path_len, output_dim):
        # Base network is shared
        # Prob networks is list of head networks that'll each output a categorical
        super().__init__(prob_network, output_dim)
        self.recurrent_network = recurrent_network
        self.path_len=path_len
        self.modules = [self.recurrent_network, self.prob_network]
        self.apply(xavier_init)

    def init_input(self, bs):
        # Return one hot for each cat
        return Variable(torch.zeros(bs, self.output_dim))

    def step(self, x):
        # x is (1, bs, input_dim)
        # Return (bs, sum(cat_sizes)) one hot and (bs, len(cat_sizes)) idxs
        out = self.recurrent_network(x).squeeze()
        # probs = [f(out) for f in self.prob_networks]
        # argmax = [torch.max(prob, -1)[1] for prob in probs]
        # return torch.cat(probs, -1), torch.stack(argmax, -1)
        probs = self.prob_network(out)
        argmax = torch.max(probs, -1)[1]
        return probs, argmax


    def forward(self, z, initial_input=None):
        # z is (bs, latent_dim)
        bs = z.size()[0]
        self.recurrent_network.init_hidden(bs)
        if initial_input is None:
            initial_input = self.init_input(bs)

        z = z.unsqueeze(0)  # (1, bs, latent_dim)
        x = initial_input.unsqueeze(0)  # (1, bs, sum(cat_sizes))

        probs, argmaxs = [], []
        for s in range(self.path_len):
            x = torch.cat([x, z], -1)
            prob, argmax = self.step(x)
            probs.append(prob)
            argmaxs.append(argmax)
            x = prob.unsqueeze(0)


        probs = torch.stack(probs, 1)  # (bs, path_len, sum(cat_sizes))
        argmaxs = torch.stack(argmaxs, 1) # (bs, path_len, len(cat_sizes))
        onehot = np_to_var(np.eye(self.output_dim)[get_numpy(argmaxs)])
        dist = RecurrentCategorical(probs, self.path_len, onehot)
        return dist

    def recurrent(self):
        return True

class RecurrentMultiCategoricalNetwork(ModuleContainer):
    def __init__(self, recurrent_network, prob_networks, cat_sizes, path_len, output_dim):
        # Base network is shared
        # Prob networks is list of head networks that'll each output a categorical
        super().__init__()
        self.recurrent_network = recurrent_network
        self.prob_networks = prob_networks
        self.path_len=path_len
        self.cat_sizes = cat_sizes
        self.output_dim = output_dim
        self.modules = [self.recurrent_network] + self.prob_networks
        self.apply(xavier_init)

    def init_input(self, bs):
        # Return one hot for each cat
        return Variable(torch.zeros(bs, self.output_dim))

    def step(self, x):
        # x is (1, bs, input_dim)
        # Return (bs, sum(cat_sizes)) one hot and (bs, len(cat_sizes)) idxs
        out = self.recurrent_network(x).squeeze()
        probs = [f(out) for f in self.prob_networks]
        argmax = [torch.max(prob, -1)[1] for prob in probs]
        return torch.cat(probs, -1), torch.stack(argmax, -1)


    def forward(self, z, initial_input=None):
        # z is (bs, latent_dim)
        bs = z.size()[0]
        self.recurrent_network.init_hidden(bs)
        if initial_input is None:
            initial_input = self.init_input(bs)

        z = z.unsqueeze(0)  # (1, bs, latent_dim)
        x = initial_input.unsqueeze(0)  # (1, bs, sum(cat_sizes))

        probs, argmaxs = [], []
        for s in range(self.path_len):
            x = torch.cat([x, z], -1)
            prob, argmax = self.step(x)
            probs.append(prob)
            argmaxs.append(argmax)
            x = prob.unsqueeze(0)


        probs = torch.stack(probs, 1)  # (bs, path_len, sum(cat_sizes))
        argmaxs = torch.stack(argmaxs, 1) # (bs, path_len, len(cat_sizes))
        dist = RecurrentMultiCategorical(prob=probs.view(bs, -1), path_len=self.path_len,
                                         cat_sizes=self.cat_sizes, mle=argmaxs)
        return dist

    def recurrent(self):
        return True