import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from traj2vec.algos.batch_polopt import BatchPolopt
from traj2vec.utils.torch_utils import from_numpy, Variable, get_numpy, LongTensor
import traj2vec.utils.logger as logger

class PPO(BatchPolopt):
    def __init__(
            self,
            env,
            env_name,
            policy,
            baseline,
            ppo_batch_size=64,
            epoch=10,
            clip_param=0.2,
            step_size=3e-4,
            optimizer=None,
            reset_args=None,
            **kwargs):
        if optimizer is None:
            optimizer = optim.Adam(policy.get_params() + list(baseline.network.parameters()), lr=step_size)
        self.ppo_batch_size = ppo_batch_size
        self.optimizer = optimizer
        self.epoch = epoch
        self.clip_param = clip_param
        self.reset_args = reset_args

        super(PPO, self).__init__(env=env, env_name=env_name, policy=policy, baseline=baseline, **kwargs)
        self.fit_baseline = False # Don't do it during process samples. Will happen during optimize

    def obtain_samples(self, itr):
        if self.reset_args is not None:
            return self.sampler.obtain_samples(self.batch_size, self.max_path_length, volatile=True, reset_args=self.reset_args)
        return self.sampler.obtain_samples(self.batch_size, self.max_path_length, volatile=True)
    def optimize_policy(self, itr, samples_data, add_input_fn=None, add_input_input=None, add_loss_fn=None, print=True):

        advantages = from_numpy(samples_data['discount_adv'].astype(np.float32))
        n_traj = samples_data['obs'].shape[0]
        n_obs = n_traj * self.max_path_length
        #add_input_obs = from_numpy(samples_data['obs'][:, :, :self.obs_dim].astype(np.float32)).view(n_traj, -1)
        if add_input_fn is not None:
            obs = from_numpy(samples_data['obs'][:, :self.max_path_length, :self.obs_dim].astype(np.float32)).view(n_obs, -1)
        else:
            obs = from_numpy(samples_data['obs'][:, :self.max_path_length, :].astype(np.float32)).view(n_obs, -1)

        #obs = from_numpy(samples_data['obs'][:, :self.max_path_length, :].astype(np.float32)).view(n_obs, -1)

        actions = samples_data['actions'].view(n_obs, -1).data
        returns = from_numpy(samples_data['discount_returns'].copy()).view(-1, 1).float()
        old_action_log_probs = samples_data['log_prob'].view(n_obs, -1).data
        states = samples_data['states'].view(samples_data['states'].size()[0], n_obs, -1) if self.policy.recurrent() else None

        for epoch_itr in range(self.epoch):
            sampler = BatchSampler(SubsetRandomSampler(range(n_obs)),
                                   self.ppo_batch_size, drop_last=False)
            for indices in sampler:
                indices = LongTensor(indices)
                obs_batch = Variable(obs[indices])
                actions_batch = actions[indices]
                return_batch = returns[indices]
                old_action_log_probs_batch = old_action_log_probs[indices]
                if states is not None:
                    self.policy.set_state(Variable(states[:, indices]))

                if add_input_fn is not None:
                    add_input_dist = add_input_fn(Variable(add_input_input))
                    add_input = add_input_dist.sample()
                    add_input_rep = torch.unsqueeze(add_input, 1).repeat(1, self.max_path_length, 1).view(n_obs, -1)
                    #add_input_batch = add_input[indices/add_input.size()[0]]
                    add_input_batch = add_input_rep[indices]
                    obs_batch = torch.cat([obs_batch, add_input_batch], -1)




                values = self.baseline.forward(obs_batch.detach())
                action_dist = self.policy.forward(obs_batch)
                action_log_probs = action_dist.log_likelihood(Variable(actions_batch)).unsqueeze(-1)
                dist_entropy = action_dist.entropy().mean()

                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                adv_targ = Variable(advantages.view(-1, 1)[indices])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                self.optimizer.zero_grad()

                total_loss = (value_loss + action_loss - dist_entropy * self.entropy_bonus)
                if add_loss_fn is not None:
                    total_loss += add_loss_fn(add_input_dist, add_input, add_input_input)
                total_loss.backward()
                self.optimizer.step()
            if print:
                stats = {'total loss': get_numpy(total_loss)[0],
                         'action loss': get_numpy(action_loss)[0],
                         'value loss': get_numpy(value_loss)[0],
                         'entropy' : get_numpy(dist_entropy)[0]}
                with logger.prefix('Train PPO itr %d epoch itr %d | ' %(itr, epoch_itr)):
                    self.print_diagnostics(stats)



        return total_loss