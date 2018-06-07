import torch
from traj2vec.math.distributions.distribution import Distribution


class Mixed(Distribution):
    def __init__(self, gaussian, cat, path_len):
        self.gaussian = gaussian
        self.cat = cat
        self.path_len = path_len
        self.g_dim = self.gaussian.dim
        self.c_dim = self.cat.dim
        self._mle = None
        self.log_var = self.gaussian.log_var


    def log_likelihood(self, x):
        # Assumes x is batch size by feature dim
        # Returns log_likehood for each point in batch
        bs = x.size()[0]
        x1, x2 = x.view(bs, self.path_len, -1).split(self.g_dim//self.path_len, -1)
        gll = self.gaussian.log_likelihood(x1.contiguous().view(bs, -1))
        cll = torch.log((x2 * self.cat.prob.view(bs, self.path_len, -1)).sum(-1) + 1e-8).sum(-1)
        #print(get_numpy(gll.mean())[0], get_numpy(cll.mean())[0])
        # TODO find better way to scale categorical loss
        return gll + 100 * cll

    #
    def reshape(self, new_shape):
        return Mixed(self.gaussian.reshape(new_shape), self.cat.reshape(new_shape), self.path_len)
        #return Normal(self.mean.view(*new_shape), self.log_var.view(*new_shape))

    @property
    def mle(self):
        if self._mle is None:
            bs = self.gaussian.mle.size()[0]
            shape_3d = (bs, self.path_len, -1)
            self._mle = torch.cat([self.gaussian.mle.view(shape_3d), self.cat.mle.view(shape_3d)], -1).view(bs, -1)
            #self._mle = torch.cat([self.gaussian.mle.view(shape_3d), self.cat.prob.view(shape_3d)], -1).view(bs, -1)
        return self._mle

    def sample(self):
        bs = self.gaussian.mean.size()[0]
        g1 = self.gaussian.sample().view(bs, self.path_len, -1)
        c1 = self.cat.sample()
        return torch.cat([g1, c1], -1)