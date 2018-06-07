import numpy as np
import torch
from traj2vec.math.distributions.categorical import RecurrentCategorical
from traj2vec.math.distributions.mixed import Mixed
from traj2vec.math.distributions.normal import Normal
from traj2vec.models.containers.module_container import ModuleContainer
from traj2vec.nn.weight_inits import xavier_init
from traj2vec.utils.torch_utils import np_to_var, Variable, get_numpy


class MixedRecurrentNetwork(ModuleContainer):
    def __init__(self, mean_network, log_var_network, prob_network, recurrent_network, path_len, output_dim,
                 gaussian_output_dim, cat_output_dim,
                 min_var=1e-4):
        super().__init__()
        self.mean_network = mean_network
        self.log_var_network = log_var_network
        self.prob_network = prob_network
        self.recurrent_network = recurrent_network
        self.modules = [self.mean_network, self.log_var_network, self.recurrent_network, self.prob_network]
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))
        self.apply(xavier_init)

        self.gaussian_output_dim = gaussian_output_dim
        self.cat_output_dim = cat_output_dim # Onehot size
        self.output_dim = output_dim
        self.path_len = path_len

    def init_input(self, bs):
        return Variable(torch.zeros(bs, self.output_dim))

    def step(self, x):
        # Torch uses (path_len, bs, input_dim) for recurrent input
        # Return (1, bs, output_dim)
        assert x.size()[0] == 1, "Path len must be 1"
        out = self.recurrent_network(x).squeeze()
        mean, log_var, prob = self.mean_network(out), self.log_var_network(out), self.prob_network(out)
        #log_var = torch.max(self.min_log_var, log_var)
        return mean.unsqueeze(0), log_var.unsqueeze(0), prob.unsqueeze(0)

    def forward(self, z, initial_input=None):
        # z is (bs, latent_dim)
        # Initial input is initial obs (bs, obs_dim)
        bs = z.size()[0]
        self.recurrent_network.init_hidden(bs)
        if initial_input is None:
            initial_input = self.init_input(bs)

        z = z.unsqueeze(0) # (1, bs, latent_dim)
        initial_input = initial_input.unsqueeze(0) # (1, bs, obs_dim)

        means, log_vars, probs, onehots = [], [], [], []
        x = initial_input

        for s in range(self.path_len):
            x = torch.cat([x, z], -1)
            mean, log_var, prob = self.step(x)
            onehot = np.zeros(prob.size()[1:])
            onehot[np.arange(0, bs), get_numpy(torch.max(prob.squeeze(0), -1)[1]).astype(np.int32)] = 1
            onehot = np_to_var(onehot).unsqueeze(0)
            x = torch.cat([mean, onehot], -1)
            #x = Variable(torch.randn(mean.size())) * torch.exp(log_var) + mean
            means.append(mean.squeeze(dim=0))
            log_vars.append(log_var.squeeze(dim=0))
            probs.append(prob.squeeze(dim=0))
            onehots.append(onehot.squeeze(dim=0))

        means = torch.stack(means, 1).view(bs, -1)
        log_vars = torch.stack(log_vars, 1).view(bs, -1)
        probs = torch.stack(probs, 1).view(bs, -1)
        onehots = torch.stack(onehots, 1).view(bs, -1)

        gauss_dist = Normal(means, log_var=log_vars)
        cat_dist = RecurrentCategorical(probs, self.path_len, onehots)

        return Mixed(gauss_dist, cat_dist, self.path_len)

    def recurrent(self):
        return True


