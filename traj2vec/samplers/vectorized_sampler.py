import numpy as np
import torch
import traj2vec.utils.logger as logger
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from traj2vec.envs.env_utils import make_env
from traj2vec.samplers.sampler import Sampler
from traj2vec.utils.torch_utils import Variable, from_numpy, get_numpy


class VectorizedSampler(Sampler):
    def __init__(self, env_name, n_envs, envs=None, random_action_p=0, ego=False, egoidx=None, **kwargs):
        super().__init__(**kwargs)
        self.n_envs = n_envs
        if envs is None:
            if env_name is None:
                envs = [kwargs['env']() for i in range(n_envs)]
            else:
                envs = [make_env(env_name, int(np.random.randint(0, 10000, 1)), i, logger.get_snapshot_dir())() for i in range(n_envs)]
        self.envs = envs
        self.random_action_p = random_action_p
        self.ego = ego
        self.egoidx = egoidx
        #self.envs = [copy.deepcopy(self.env) for _ in range(self.n_envs)]

    def reset_envs(self, reset_args):
        if reset_args is not None:
            return np.stack([env.reset(args) for args, env in zip(reset_args, self.envs)])
        return np.stack([env.reset() for env in self.envs])

    def step_envs(self, action):
        # (obs, reward, done, env_info)
        step_env = [env.step(action[i, :]) for i, env in enumerate(self.envs)]
        obs = np.stack([x[0] for x in step_env], 0)
        reward = np.array([x[1] for x in step_env])
        done = np.array([x[2] for x in step_env])
        env_info = tuple([x[3] for x in step_env])
        return obs, reward, done, env_info

    def rollout(self, max_path_length, add_input=None, reset_args=None, volatile=False):
        sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[], states=[])
        obs = self.reset_envs(reset_args)
        self.policy.reset(obs.shape[0])

        for s in range(max_path_length):
            state = self.policy.get_state().data if self.policy.recurrent() else None

            if self.ego:
                obs_ego = obs.copy()
                obs_ego[:, self.egoidx] -= reset_args[:, self.egoidx]
                policy_input = Variable(from_numpy(obs_ego).float(), volatile=volatile)
            else:
                policy_input = Variable(from_numpy(obs).float(), volatile=volatile)

            if add_input is not None:
                policy_input = torch.cat([policy_input, add_input], -1)
            action_dist = self.policy.forward(policy_input)

            action = action_dist.sample()

            if self.random_action_p > 0:
                flip = np.random.binomial(1, self.random_action_p, size=len(obs))
                if flip.sum() > 0:
                    random_act = np.random.randint(0, self.policy.output_dim, size=flip.sum())
                    action[from_numpy(flip).byte()] = from_numpy(random_act)

            next_obs, rewards, done, info = self.step_envs(get_numpy(action))
            #env_step = self.step_envs(get_numpy(action))
            #next_obs = [x[0] for x in env_step]
            sd['obs'].append(obs)
            sd['rewards'].append(rewards)
            sd['actions'].append(action)
            sd['action_dist_lst'].append(action_dist)
            sd['states'].append(state)
            obs = next_obs
        # Append last obs
        sd['obs'].append(obs)
        sd['obs'] = np.stack(sd['obs'], 1) # (bs, max_path_length, obs_dim)
        #import pdb; pdb.set_trace()
        sd['states'] = torch.stack(sd['states'], 2) if self.policy.recurrent() else None
        sd['rewards'] = np.stack(sd['rewards'], 1) # (bs, max_path_length)
        sd['actions'] = torch.stack(sd['actions'], 1)

        sd['action_dist'] = sd['action_dist_lst'][0].combine(sd['action_dist_lst'],
                torch.stack, axis=1)

        return sd


    def obtain_samples(self, batch_size, max_path_length, add_input=None, volatile=False, reset_args=None):

        num_trajs = batch_size // max_path_length
        num_rollouts = int(np.ceil(num_trajs / self.n_envs))
        sd = dict(obs=[], rewards=[], actions=[], action_dist=[], states=[])
        add_input_slice = None
        reset_args_slice = None
        for n_r in range(num_rollouts):
            if add_input is not None:
                add_input_slice = add_input[n_r*self.n_envs : (n_r+1) * self.n_envs]
            if reset_args is not None:
                reset_args_slice = reset_args[n_r*self.n_envs : (n_r+1) * self.n_envs]
            rollout_data = self.rollout(max_path_length, add_input_slice, reset_args_slice, volatile)
            for k in sd.keys():
                sd[k].append(rollout_data[k])

        sd['obs'] = np.concatenate(sd['obs'])
        sd['states'] = torch.cat(sd['states'], 1) if self.policy.recurrent() else None
        sd['rewards'] = np.concatenate(sd['rewards'])
        sd['actions'] = torch.cat(sd['actions'])
        sd['action_dist'] = sd['action_dist'][0].combine(sd['action_dist'],
                                        torch.cat, axis=0)

        return sd

class ParVectorizedSampler(VectorizedSampler):
    """Parallel version of Vectorized Sampler using OpenAI Baselines SubprocVecEnv"""
    def __init__(self, env_name, n_envs=10, envs=None, random_action_p=0, **kwargs):
        Sampler.__init__(self, **kwargs)
        self.n_envs = 20
        if envs is None:
            envs = SubprocVecEnv([make_env(env_name, 1, i, logger.get_snapshot_dir()) for i in range(self.n_envs)])
        self.envs = envs
        self.random_action_p = random_action_p
        #self.envs = [copy.deepcopy(self.env) for _ in range(self.n_envs)]

    def reset_envs(self, reset_args):
        # TODO Fix SubprocVecEnv does not accept args
        return self.envs.reset()

    def step_envs(self, action):
        # (obs, reward, done, env_info)
        return self.envs.step(action)

