import time
from collections import OrderedDict

import numpy as np
import torch
import traj2vec.utils.logger as logger
from traj2vec.samplers.rollout import rollout
from traj2vec.samplers.vectorized_sampler import VectorizedSampler
from traj2vec.utils.torch_utils import gpu_enabled, get_numpy, FloatTensor, DoubleTensor


class BatchPolopt:
    def __init__(
            self,
            env,
            env_name,
            policy,
            baseline,
            obs_dim,
            action_dim,
            save_step=20,
            plot=False,
            plot_itr_threshold=0,
            plot_every=10,
            n_itr=500,
            start_itr=0,
            batch_size=1,
            max_path_length=500,
            discount=0.99,
            sampler=None,
            n_vectorized_envs=None,
            center_adv=True,
            fit_baseline=True,
            use_gae=False,
            gae_tau=0.95,
            entropy_bonus=0,
            alter_sd_fn=None,
            alter_sd_args=None,
            env_obj=None,
    ):
        self.env = env
        if env_obj is None:
            self.env_obj = env()
        else:
            self.env_obj = env_obj
        self.env_name = env_name
        self.policy = policy
        self.baseline = baseline
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.save_step = save_step
        self.plot = plot
        self.plot_itr_threshold = plot_itr_threshold
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.center_adv = center_adv
        self.fit_baseline = fit_baseline
        self.alter_sd_fn = alter_sd_fn
        self.alter_sd_args = alter_sd_args
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.entropy_bonus = entropy_bonus
        self.plot_every = plot_every

        if sampler is None:
            if n_vectorized_envs is None:
                n_vectorized_envs = max(1, int(np.ceil(batch_size / max_path_length)))

            sampler = VectorizedSampler(env_name=env_name, env=env, policy=policy, n_envs=n_vectorized_envs)

        self.sampler = sampler

        if gpu_enabled():
            self.policy.cuda()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(self.batch_size, self.max_path_length)

    def process_samples(self, itr, sd, augment_obs=None):
        # rewards is (bs, path_len)
        # actions is (bs, path_len, acton_dim)
        if augment_obs is not None:
            sd['obs'] = np.concatenate([sd['obs'], np.repeat(np.expand_dims(augment_obs, 1), sd['obs'].shape[1], 1)], -1)
        sd['values'] = get_numpy(self.baseline.predict(sd['obs'].astype(np.float32)))

        sd['rewards'] = sd['rewards'].astype(np.float32)
        path_len = sd['rewards'].shape[-1]


        returns = np.zeros((sd['rewards'].shape[0], sd['rewards'].shape[1] + 1))
        returns[:, -2] = sd['values'][:, -1]
        rewards = sd['rewards']
        if self.use_gae:
            gae = 0
            for step in reversed(range(rewards.shape[1])):
                mask = 1 if step != rewards.shape[1] - 1 else 0
                delta = rewards[:, step] + self.discount * sd['values'][:, step + 1] * mask - sd['values'][:, step]
                gae = delta + self.discount * self.gae_tau * mask * gae
                returns[:, step] = gae + sd['values'][:, step]
                #import pdb; pdb.set_trace()
        else:
            for step in reversed(range(rewards.shape[1])):
                returns[:, step] = returns[:, step + 1] * self.discount + rewards[:, step]

        sd['returns'] = np.cumsum(sd['rewards'][:, ::-1], axis=-1)[:, ::-1]
        sd['discount_returns'] = returns[:, :-1]
        sd['actions'] = sd['actions'].detach()
        sd['log_prob'] = sd['action_dist'].log_likelihood(sd['actions'])
        sd['entropy'] = sd['action_dist'].entropy()

        # logger.log('Fitting Baseline')
        #sd['adv'] = sd['returns'] - sd['values']
        if self.fit_baseline:
           self.baseline.fit(sd['obs'][:, :-1], sd['discount_returns'])

        if hasattr(self.policy, 'obs_filter') and self.policy.obs_filter is not None:
            self.policy.obs_filter.update(DoubleTensor(sd['obs'][:, :-1].reshape((-1, self.policy.obs_filter.shape))))

        if sd['values'].shape[1] > 1:
            sd['discount_adv'] = sd['discount_returns'] - sd['values'][:, :-1]
        else:
            sd['discount_adv'] = sd['discount_returns'] - sd['values']

        if self.center_adv:
            sd['discount_adv'] = (sd['discount_adv'] - sd['discount_adv'].mean()) / (
                sd['discount_adv'].std() + 1e-5
            )

        sd['stats'] = OrderedDict([
            ('Mean Return', sd['returns'][:, 0].mean()),
            ('Min Return', sd['returns'][:, 0].min()),
            ('Max Return', sd['returns'][:, 0].max()),
            ('Var Return', sd['returns'][:, 0].var()),
            ('Entropy', get_numpy(sd['entropy'].mean())[0]),
            #('Policy loss', -(get_numpy(sd['log_prob']) * sd['discount_adv']).sum(-1).mean()),
        ])

    def save(self, snapshot_dir, itr):
        import os
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        torch.save(self.policy.state_dict_lst(), snapshot_dir + '/policy_%d.pkl' %itr)

    def load(self, snapshot_dir, itr):
        self.policy.load_state_dict(torch.load(snapshot_dir + '/policy_%d.pkl' %itr))

    def train(self):
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                sd = self.obtain_samples(itr)
                if self.alter_sd_fn is not None:
                    self.alter_sd_fn(sd, *self.alter_sd_args)
                logger.log("Processing samples...")
                self.process_samples(itr, sd)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(sd['stats'])
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, sd)
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
            if itr % self.plot_every == 0 and self.plot and itr > self.plot_itr_threshold:
                rollout(self.policy, self.env_obj, self.max_path_length, plot=True)
            if itr % self.save_step == 0 and logger.get_snapshot_dir() is not None:
                self.save(logger.get_snapshot_dir() + '/snapshots', itr)

    def print_diagnostics(self, stats):
        for k, v in stats.items():
            logger.log('%s: %f' % (k, v))

    def log_diagnostics(self, stats):
        for k, v in stats.items():
            logger.record_tabular(k, v)

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError
