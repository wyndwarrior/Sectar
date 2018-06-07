import numpy as np
import torch
import traj2vec.utils.logger as logger
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from traj2vec.envs.env_utils import make_env
from traj2vec.samplers.sampler import Sampler
from traj2vec.utils.torch_utils import Variable, from_numpy, get_numpy


class BatchSampler(Sampler):
    def __init__(self, env_name, n_envs, envs=None, random_action_p=0, **kwargs):
        super().__init__(**kwargs)
        self.n_envs = n_envs
        if envs is None:
            envs = SubprocVecEnv([make_env(env_name, 1, i, logger.get_snapshot_dir()) for i in range(n_envs)])
        self.envs = envs
        self.random_action_p = random_action_p
        #self.envs = [copy.deepcopy(self.env) for _ in range(self.n_envs)]

    def rollout(self, max_path_length, add_input=None, volatile=False):
        sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[])
        obs = self.envs.reset()
        self.policy.reset(len(obs))
        for s in range(max_path_length):
            policy_input = Variable(from_numpy(np.stack(obs)).float(), volatile=volatile)
            if add_input is not None:
                policy_input = torch.cat([policy_input, add_input], -1)
            action_dist = self.policy.forward(policy_input)

            action = action_dist.sample()

            if self.random_action_p > 0:
                flip = np.random.binomial(1, self.random_action_p, size=len(obs))
                if flip.sum() > 0:
                    random_act = np.random.randint(0, int(self.env.action_space.flat_dim), size=flip.sum())
                    action[from_numpy(flip).byte()] = from_numpy(random_act)

            next_obs, rewards, done, info = self.envs.step(get_numpy(action))

            sd['obs'].append(obs)
            sd['rewards'].append(rewards)
            sd['actions'].append(action)
            sd['action_dist_lst'].append(action_dist)
            obs = next_obs
        # Append last obs
        sd['obs'].append(obs)
        sd['obs'] = np.stack(sd['obs'], 1) # (bs, max_path_length, obs_dim)
        sd['rewards'] = np.stack(sd['rewards'], 1) # (bs, max_path_length)
        sd['actions'] = torch.stack(sd['actions'], 1)

        sd['action_dist'] = sd['action_dist_lst'][0].combine(sd['action_dist_lst'],
                torch.stack, axis=1)

        return sd


    def obtain_samples(self, batch_size, max_path_length, add_input=None, volatile=False):
        num_trajs = batch_size // max_path_length
        num_rollouts = int(np.ceil(num_trajs / self.n_envs))
        sd = dict(obs=[], rewards=[], actions=[], action_dist=[])
        add_input_slice = None
        for n_r in range(num_rollouts):
            if add_input is not None:
                add_input_slice = add_input[n_r*self.n_envs : (n_r+1) * self.n_envs]
            rollout_data = self.rollout(max_path_length, add_input_slice, volatile)
            for k in sd.keys():
                sd[k].append(rollout_data[k])

        sd['obs'] = np.concatenate(sd['obs'])
        sd['rewards'] = np.concatenate(sd['rewards'])
        sd['actions'] = torch.cat(sd['actions'])
        sd['action_dist'] = sd['action_dist'][0].combine(sd['action_dist'],
                                        torch.cat, axis=0)

        return sd

