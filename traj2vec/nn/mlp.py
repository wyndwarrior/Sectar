import torch.nn as nn
from traj2vec.nn.weight_inits import weights_init_mlp, xavier_init

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32),
                 hidden_act=nn.ReLU, final_act=None, weight_init=xavier_init):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        mlp = nn.Sequential()
        prev_size = input_dim

        for i, hidden_size in enumerate(hidden_sizes):
            mlp.add_module(name='linear %d' % i, module=nn.Linear(prev_size, hidden_size))
            mlp.add_module(name='relu %d' % i, module=hidden_act())
            prev_size = hidden_size

        mlp.add_module(name='finallayer', module=nn.Linear(hidden_sizes[-1], output_dim))

        if final_act is not None:
            mlp.add_module(name='finalact', module=final_act())

        self.network = mlp

        self.apply(weight_init)

    def forward(self, x):
        return self.network(x)
