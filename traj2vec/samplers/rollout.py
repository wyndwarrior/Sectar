import torch
from traj2vec.utils.torch_utils import np_to_var, get_numpy


def rollout(policy, env, max_path_length, add_input=None, plot=False):
    obs = env.reset()
    sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[])

    for s in range(max_path_length):
        if add_input is not None:
            policy_input = torch.cat([np_to_var(obs), add_input], -1).view(1, -1)
        else:
            policy_input = np_to_var(obs).unsqueeze(0)
        action_dist = policy.forward(policy_input)
        action = action_dist.sample()

        next_obs, reward, done, info = env.step(get_numpy(action.squeeze()))
        sd['obs'].append(obs)
        sd['rewards'].append(reward)
        sd['actions'].append(action)
        sd['action_dist_lst'].append(action_dist)
        obs = next_obs

        if plot:
            env.render()

    return sd

