import numpy as np
import torch
from traj2vec.math.distributions.distribution import Distribution
from traj2vec.utils.torch_utils import Variable


class Normal(Distribution):
    def __init__(self, mean, log_var):
        self.mean = mean
        self.dim = self.mean.size()[-1]
        self.log_var = log_var


    def log_likelihood(self, x):
        # Assumes x is batch size by feature dim
        # Returns log_likehood for each point in batch
        zs = (x - self.mean) / torch.exp(self.log_var)

        return -torch.sum(self.log_var, -1) - \
               0.5 * torch.sum(torch.pow(zs, 2), -1) -\
               0.5 * self.dim * np.log(2*np.pi)

    def log_likelihood_ratio(self, x, new_dist):
        ll_new = new_dist.log_likelihood(x)
        ll_old = self.log_likelihood(x)
        return torch.exp(ll_new - ll_old)

    def entropy(self):
        return torch.sum(self.log_var + np.log(np.sqrt(2 * np.pi * np.e)), -1)

    def kl(self, new_dist):
        # Compute D_KL(self || dist)
        old_var = torch.exp(self.log_var)
        new_var = torch.exp(new_dist.log_var)
        numerator = torch.pow(self.mean - new_dist.mean, 2) + \
                    torch.pow(old_var, 2) - torch.pow(new_var, 2)
        denominator = 2 * torch.pow(new_var, 2) + 1e-8


        return torch.sum(
            numerator / denominator + new_dist.log_var - self.log_var, -1
        )


    def sample(self, deterministic=False):
        if deterministic:
            return self.mean
        else:
            return Variable(torch.randn(self.mean.size())) * torch.exp(self.log_var) + self.mean


    def combine(self, dist_lst, func=torch.stack, axis=0):
        self.mean = func([dist.mean for dist in dist_lst], axis)
        #self.var = func([dist.var for dist in dist_lst], axis)
        self.log_var = func([dist.log_var for dist in dist_lst], axis)
        self.dim = self.mean.size()[-1]
        return self

    def detach(self):
        return Normal(self.mean.detach(), self.log_var.detach())

    def reshape(self, new_shape):
        return Normal(self.mean.view(*new_shape), self.log_var.view(*new_shape))

    @property
    def mle(self):
        return self.mean