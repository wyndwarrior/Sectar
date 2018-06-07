import torch.optim as optim
from traj2vec.algos.batch_polopt import BatchPolopt
from traj2vec.utils.torch_utils import np_to_var


class VPG(BatchPolopt):
    def __init__(
            self,
            env,
            env_name,
            policy,
            baseline,
            step_size=0.01,
            optimizer=None,
            **kwargs):
        if optimizer is None:
            optimizer = optim.Adam(policy.get_params(), lr=step_size)
        self.optimizer = optimizer
        super(VPG, self).__init__(env=env, env_name=env_name, policy=policy, baseline=baseline, **kwargs)

    def loss(self, sd):
        loss = -(sd['log_prob'] * np_to_var(sd['discount_adv'])[:, :self.max_path_length]) - self.entropy_bonus * sd['entropy']
        return loss.sum(-1).mean(0)

    def optimize_policy(self, itr, samples_data, custom_loss_fn=None):
        self.optimizer.zero_grad()
        if custom_loss_fn is not None:
            loss = custom_loss_fn(samples_data)
        else:
            loss = self.loss(samples_data)

        loss.backward()
        self.optimizer.step()
        return loss
