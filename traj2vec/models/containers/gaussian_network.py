from traj2vec.math.distributions.normal import Normal
from traj2vec.models.containers.module_container import ModuleContainer
from traj2vec.utils.torch_utils import np_to_var, Variable
from traj2vec.nn.weight_inits import weights_init_mlp
#from torch.autograd import Variable
import numpy as np
import torch
# from torch.autograd import Variable
from torch import nn
from traj2vec.nn.rnn import RNN
import numpy as np
import torch
from traj2vec.math.distributions.normal import Normal
from traj2vec.models.containers.module_container import ModuleContainer
from traj2vec.nn.weight_inits import weights_init_mlp, xavier_init
from traj2vec.utils.torch_utils import np_to_var, Variable


class GaussianNetwork(ModuleContainer):
    def __init__(self, mean_network, log_var_network, init=xavier_init, scale_final=False,
                 min_var=1e-4, obs_filter=None):
        super().__init__()
        self.mean_network = mean_network
        self.log_var_network = log_var_network
        self.modules = [self.mean_network, self.log_var_network]
        self.obs_filter = obs_filter
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

        self.apply(init)
        # self.apply(weights_init_mlp)
        if scale_final:
            if hasattr(self.mean_network, 'network'):
                self.mean_network.network.finallayer.weight.data.mul_(0.01)


    def forward(self, x):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        mean, log_var = self.mean_network(x), self.log_var_network(x)
        log_var = torch.max(self.min_log_var, log_var)
        # TODO Limit log var
        dist = Normal(mean=mean, log_var=log_var)
        return dist

    def recurrent(self):
        return False

    def reset(self, bs):
        pass


class GaussianLSTMPolicy(ModuleContainer):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, log_var_network, init=xavier_init, scale_final=False,
                 min_var=1e-4, obs_filter=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.rnn = RNN(self.lstm, hidden_dim)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.log_var_network = log_var_network
        self.modules = [self.rnn, self.linear, self.log_var_network]

        self.obs_filter = obs_filter
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

        self.apply(init)
        # self.apply(weights_init_mlp)
        if scale_final:
            if hasattr(self.mean_network, 'network'):
                self.mean_network.network.finallayer.weight.data.mul_(0.01)


    def forward(self, x):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        lstm_out = self.rnn.forward(x)
        lstm_reshape = lstm_out.view((-1, self.hidden_dim))
        mean = self.softmax(self.linear(lstm_reshape))

        log_var = self.log_var_network(x.contiguous().view((-1, x.shape[-1])))
        log_var = torch.max(self.min_log_var, log_var)
        # TODO Limit log var
        dist = Normal(mean=mean, log_var=log_var)
        return dist


    def reset(self, bs):
        self.rnn.init_hidden(bs)


    def set_state(self, state):
        self.rnn.set_state(state)

    def get_state(self):
        return self.rnn.get_state()

    def recurrent(self):
        return True

class GaussianBidirectionalNetwork(GaussianNetwork):

    def __init__(self, input_dim, hidden_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        # self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True)
        self.rnn = RNN(self.lstm, hidden_dim)
        self.modules.extend([self.rnn])
        # self.linear = nn.Linear(hidden_dim * 2, output_dim)
    def reset(self, x):
        self.rnn.init_hidden(x.size()[1])
        
    def forward(self, x):
        self.reset(x)
        lstm_out = self.rnn.forward(x)
        lstm_mean = torch.mean(lstm_out, dim=0)
        # output = self.linear(lstm_mean)
        mean, log_var = self.mean_network(lstm_mean), self.log_var_network(lstm_mean)
        dist = Normal(mean, log_var=log_var)
        return dist

    def recurrent(self):
        return True
class GaussianRecurrentNetwork(GaussianNetwork):
    def __init__(self, recurrent_network, path_len, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.recurrent_network = recurrent_network
        self.modules.extend([recurrent_network])
        self.output_dim = output_dim
        self.path_len = path_len

    def init_input(self, bs):
        return Variable(torch.zeros(bs, self.output_dim))

    def step(self, x):
        # Torch uses (path_len, bs, input_dim) for recurrent input
        # Return (1, bs, output_dim)
        assert x.size()[0] == 1, "Path len must be 1"
        out = self.recurrent_network(x).squeeze(0)
        mean, log_var = self.mean_network(out), self.log_var_network(out)
        #log_var = torch.max(self.min_log_var, log_var)
        return mean.unsqueeze(0), log_var.unsqueeze(0)

    def forward(self, z, initial_input=None):
        # z is (bs, latent_dim)
        # Initial input is initial obs (bs, obs_dim)
        bs = z.size()[0]
        self.recurrent_network.init_hidden(bs)
        if initial_input is None:
            initial_input = self.init_input(bs)

        z = z.unsqueeze(0) # (1, bs, latent_dim)
        initial_input = initial_input.unsqueeze(0) # (1, bs, obs_dim)

        means, log_vars = [], []
        x = initial_input

        for s in range(self.path_len):
            x = torch.cat([x, z], -1)
            mean, log_var = self.step(x)
            x = mean
            #x = Variable(torch.randn(mean.size())) * torch.exp(log_var) + mean
            means.append(mean.squeeze(dim=0))
            log_vars.append(log_var.squeeze(dim=0))

        means = torch.stack(means, 1).view(bs, -1)
        log_vars = torch.stack(log_vars, 1).view(bs, -1)
        dist = Normal(means, log_var=log_vars)
        return dist

    def recurrent(self):
        return True

class GaussianRecurrentPolicy(GaussianNetwork):
    def __init__(self, recurrent_network, path_len, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.recurrent_network = recurrent_network
        self.modules.extend([recurrent_network])
        self.output_dim = output_dim
        self.path_len = path_len

    def init_input(self, bs):
        return Variable(torch.zeros(bs, self.output_dim))

    def forward(self, x):
        # Torch uses (path_len, bs, input_dim) for recurrent input
        # path_len must be 1
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        assert x.size()[0] == 1, "Path len must be 1"
        out = self.recurrent_network(x).squeeze()
        mean, log_var = self.mean_network(out), self.log_var_network(out)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean, log_var=log_var)
        return dist

    def reset(self, bs):
        self.recurrent_network.init_hidden(bs)


class AggregateGaussianRecurrentNetwork(GaussianNetwork):
    """
    Given time series data, aggregates outputs into single vector compared to GaussianRNN
    """
    def __init__(self, recurrent_network, path_len, step_input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.recurrent_network = recurrent_network
        self.modules.extend([recurrent_network])
        self.output_dim = output_dim
        self.path_len = path_len
        self.step_input_dim = step_input_dim

    def init_input(self, bs):
        return Variable(torch.zeros(bs, self.output_dim))

    def forward(self, x):
        # x is (bs, step_input_dim * path_len)
        bs = x.size()[0]
        self.recurrent_network.init_hidden(bs)
        x = x.view(bs, self.path_len, self.step_input_dim).transpose(0, 1)
        # Reshape into (path_len, bs, step_input_dim)
        out = self.recurrent_network(x)
        # Aggregate outputs by averaging rnn outputs, then feed through mlp
        #import pdb; pdb.set_trace()
        # Out is (path_len*num_layers, bs, output_dim * (1 + bidirect))
        out = out[:, :, -self.recurrent_network.hidden_dim:].mean(0)

        mean = self.mean_network(out)
        log_var = self.log_var_network(out)
        dist = Normal(mean, log_var=log_var)
        return dist

    def recurrent(self):
        return True