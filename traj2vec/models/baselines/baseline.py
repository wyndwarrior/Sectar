import torch
from traj2vec.utils.torch_utils import Variable


class Baseline:
    """
    Value function
    """

    def fit(self, obs, returns):
        pass

    def forward(self, obs):
        pass

class ZeroBaseline(Baseline):
    def forward(self, obs_np):
        #import pdb; pdb.set_trace()
        bs, obs_dim = obs_np.size()
        return Variable(torch.zeros(bs))


    def predict(self, obs_np):
        bs, path_len, obs_dim = obs_np.shape
        return Variable(torch.zeros(bs, path_len))