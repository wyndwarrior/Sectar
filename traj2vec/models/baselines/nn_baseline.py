import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from traj2vec.models.baselines.baseline import Baseline
from traj2vec.nn.weight_inits import xavier_init
from traj2vec.utils.torch_utils import Variable, from_numpy, gpu_enabled, get_numpy, np_to_var


class NNBaseline(Baseline):
    def __init__(self, network, batch_size=128, max_epochs=20, optimizer=optim.Adam):
        self.network = network
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optimizer = optimizer(self.network.parameters())
        if gpu_enabled():
            self.network.cuda()

    def fit(self, obs_np, returns_np):
        self.network.apply(xavier_init)
        bs, path_len, obs_dim = obs_np.shape

        obs = from_numpy(obs_np.reshape(-1, obs_dim).astype(np.float32))
        returns = from_numpy(returns_np.reshape(-1).astype(np.float32))

        dataloader = DataLoader(TensorDataset(obs, returns), batch_size=self.batch_size,
                                 shuffle=True)
        for epoch in range(self.max_epochs):
            for x, y in dataloader:
                self.optimizer.zero_grad()
                x = Variable(x)
                y = Variable(y).float().view(-1, 1)
                loss = (self.network(x) - y).pow(2).mean()
                loss.backward()
                self.optimizer.step()
        print('loss %f' % get_numpy(loss)[0])

    def forward(self, obs):
        return self.network(obs)

    def predict(self, obs_np):
        bs, path_len, obs_dim = obs_np.shape

        obs = np_to_var(obs_np.reshape(-1, obs_dim).astype(np.float32))

        return self.network(obs).view(-1, path_len)