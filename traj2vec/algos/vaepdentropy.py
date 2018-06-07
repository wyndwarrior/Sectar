import os
from collections import OrderedDict

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import traj2vec.utils.logger as logger
from traj2vec.envs.env_utils import make_env
from traj2vec.samplers.rollout import rollout
from traj2vec.samplers.vectorized_sampler import VectorizedSampler, ParVectorizedSampler
from traj2vec.utils.plot_utils import plot_traj_sets
from traj2vec.utils.plot_utils import record_fig
from traj2vec.utils.torch_utils import gpu_enabled, from_numpy, Variable, np_to_var, get_numpy, FloatTensor
from traj2vec.utils.logger_utils import record_tabular
import torch.nn as nn
from collections import deque
from traj2vec.models.containers.module_container import ModuleContainer
from traj2vec.nn.weight_inits import xavier_init
from traj2vec.nn.rnn import RNN
# from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from traj2vec.models.containers.categorical_network import CategoricalNetwork, RecurrentCategoricalNetwork, RecurrentCategoricalPolicy
from traj2vec.nn.mlp import MLP
from torch.optim import Adam
from traj2vec.utils.torch_utils import set_gpu_mode
# from traj2vec.envs.playpen.blockplaypen import BlockPlayPen, BlockPlayPenHier
from traj2vec.nn.weight_inits import weights_init_mlp, xavier_init
from collections import defaultdict
from rllab.spaces.box import Box

class VAEPDEntropy:
    def __init__(
            self,
            env,
            env_name,
            policy,
            policy_ex,
            encoder,
            decoder,
            max_path_length,
            obs_dim,
            action_dim,
            step_dim,
            policy_algo,
            policy_ex_algo,
            dataset,
            latent_dim,
            vae,
            plan_horizon, 
            max_horizon, 
            mpc_batch,
            rand_per_mpc_step,
            mpc_explore, 
            mpc_explore_batch,
            reset_ent,
            vae_train_steps,
            mpc_explore_len,
            true_reward_scale,
            discount_factor,
            reward_fn,
            block_config,
            consis_finetuning=False,
            add_frac=1,
            batch_size=1000,
            random_sample_size=10,
            plot_size=5,
    ):
        self.reward_fn = reward_fn
        self.plan_horizon = plan_horizon
        self.max_horizon = max_horizon
        self.mpc_batch = mpc_batch
        self.rand_per_mpc_step = rand_per_mpc_step
        self.mpc_explore = mpc_explore
        self.mpc_explore_batch = mpc_explore_batch
        self.add_frac = add_frac
        self.reset_ent = reset_ent
        self.vae_train_steps = vae_train_steps
        self.mpc_explore_len = mpc_explore_len
        self.consis_finetuning = consis_finetuning
        self.true_reward_scale = true_reward_scale
        self.discount_factor = discount_factor

        self.block_config = block_config
        self.env = env
        self.env_obj = self.env()
        self.env_name = env_name
        self.policy = policy
        self.policy_ex = policy_ex
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        self.encoder = encoder
        self.decoder = decoder
        self.max_path_length = max_path_length
        self.policy_algo = policy_algo
        self.policy_ex_algo = policy_ex_algo
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.vae = vae
        self.batch_size = batch_size
        self.policy_algo = policy_algo
        self.plot_sampler = None
        self.add_sampler = None
        self.random_sample_size = random_sample_size
        self.plot_size = plot_size
        self.random_sampler = VectorizedSampler(env=self.env, env_name=env_name, policy=self.policy,
                                                n_envs=random_sample_size)
        self.sampler = VectorizedSampler(env=self.env, env_name=env_name, policy=self.policy,
                                                n_envs=random_sample_size)#ParVectorizedSampler(env=self.env, env_name=env_name, policy=self.policy, n_envs=12)
        self.sampler_mpc = VectorizedSampler(env=self.env, env_name=env_name, policy=self.policy,
                                                n_envs=mpc_batch, ego=vae.ego, egoidx=vae.egoidx)
        self.sampler_ex = VectorizedSampler(env=self.env, env_name=env_name, policy=self.policy_ex,
                                                n_envs=random_sample_size, ego=vae.ego, egoidx=vae.egoidx)#ParVectorizedSampler(env=self.env, env_name=env_name, policy=self.policy_ex, n_envs=12)
        
        self.plot_sampler = VectorizedSampler(env=self.env, env_name=env_name, policy=self.policy, n_envs=plot_size)
        self.action_space = self.env().action_space

        if gpu_enabled():
            self.policy.cuda()
            self.policy_ex.cuda()
            self.encoder.cuda()
            self.decoder.cuda()



    def train(self,
              dataset, #main training dataset
              test_dataset, #main validation dataset
              dummy_dataset, #dataset containing only data from the current iteration
              joint_training, #whether training should happen on one dataset at a time or jointly
              max_itr=1000, save_step=10, 
              train_vae_after_add=10, #how many times to train the vae after exploring
              #unused
              plot_step=0, record_stats=True, print_step=True,
              start_itr=0,add_size=0, add_interval=0
             ):

        for itr in range(1, max_itr + 1):
            if itr % save_step == 0 and logger.get_snapshot_dir() is not None:
                self.save(logger.get_snapshot_dir() + '/snapshots', itr)
                np.save(logger.get_snapshot_dir() + '/snapshots/traindata', self.dataset.train_data)

            # run mpc + explorer and collect data + stats
            stats = self.train_explorer(dataset, test_dataset, dummy_dataset, itr)
            with logger.prefix('itr #%d | ' % (itr)):
                self.vae.print_diagnostics(stats)
            record_tabular(stats, 'ex_stats.csv')

            # fit the VAE on newly collected data and replay buffer
            for vae_itr in range(train_vae_after_add):
                if joint_training:
                    vae_stats = self.train_vae_joint(dataset, dummy_dataset, test_dataset, itr, vae_itr)
                else:
                    # vae_stats = self.train_vae(dummy_dataset, None, itr, vae_itr)
                    # with logger.prefix('itr #%d vae newdata itr #%d | ' % (itr, vae_itr)):
                    #     self.vae.print_diagnostics(vae_stats)
                    # record_tabular(vae_stats, 'new_vae_stats.csv')

                    vae_stats = self.train_vae(dataset, test_dataset, itr, vae_itr)
                with logger.prefix('itr #%d vae itr #%d | ' % (itr, vae_itr)):
                    self.vae.print_diagnostics(vae_stats)
                record_tabular(vae_stats, 'vae_stats.csv')


    # this method does two things, first it runs MPC, then it runs + trains the explorer
    def train_explorer(self, dataset, test_dataset, dummy_dataset, itr):
        bs = self.batch_size

        # load fixed initial state and goals from config
        init_state = self.block_config[0]
        goals = np.array(self.block_config[1])

        # functions for computing the reward and initializing the reward state (rstate)
        # rstate is used to keep track of things such as which goal you are currently on
        reward_fn, init_rstate = self.reward_fn

        # total actual reward collected by MPC agent so far
        total_mpc_rew = np.zeros(self.mpc_batch)

        # keep track of states visited by MPC to initialize the explorer from
        all_inits = []

        # current state of mpc batche
        cur_state = np.array([init_state] * self.mpc_batch)

        # initialize the reward state for the mpc batch
        rstate = init_rstate(self.mpc_batch)

        # for visualization purposes
        mpc_preds = []
        mpc_actual = []
        mpc_span = []
        rstates = []

        # Perform MPC over max_horizon
        for T in range(self.max_horizon):
            print(T)
            
            # for goal visulization 
            rstates.append(rstate)

            # rollout imaginary trajectories using state decoder
            rollouts = self.mpc(cur_state,
                    min(self.plan_horizon, self.max_horizon - T),
                    self.mpc_explore, self.mpc_explore_batch, reward_fn, rstate)

            # get first latent of best trajectory for each batch
            np_latents = rollouts[2][:, 0]

            # rollout the first latent in simulator
            mpc_traj = self.sampler_mpc.obtain_samples(self.mpc_batch* self.max_path_length,
                self.max_path_length,
                np_to_var(np_latents),
                reset_args=cur_state)

            # update reward and reward state based on trajectory from simulator
            mpc_rew, rstate = self.eval_rewards(mpc_traj['obs'], reward_fn, rstate)

            # for logging and visualization purposes
            futures = rollouts[0] + total_mpc_rew
            total_mpc_rew += mpc_rew
            mpc_preds.append(rollouts[1][0])
            mpc_span.append(rollouts[3])
            mpc_stats = {
                'mean futures': np.mean(futures),
                'std futures' : np.std(futures),
                'mean actual' : np.mean(total_mpc_rew),
                'std actual'  : np.std(total_mpc_rew),
            }
            mpc_actual.append(mpc_traj['obs'][0])
            with logger.prefix('itr #%d mpc step #%d | ' % (itr, T)):
                self.vae.print_diagnostics(mpc_stats)
            record_tabular(mpc_stats, 'mpc_stats.csv')

            # add current state to list of states explorer can initialize from
            all_inits.append(cur_state)

            # update current state to current state of simulator
            cur_state = mpc_traj['obs'][:, -1]

        # for visualization
        for idx, (actual, pred, rs, span) in enumerate(zip(mpc_actual, mpc_preds, rstates, mpc_span)):
            dataset.plot_pd_compare([actual, pred, span[:100], span[:100, :dataset.path_len]],
                 ['actual', 'pred', 'imagined', 'singlestep'], itr, save_dir='mpc_match', name='Pred'+str(idx),
                 goals=goals, goalidx=rs[0])

        # compute reward at final state, for some tasks that care about final state reward
        final_reward, _ = reward_fn(cur_state, rstate)
        print(total_mpc_rew)
        print(final_reward)

        # randomly select states for explorer to explore
        start_states = np.concatenate(all_inits, axis=0)
        start_states = start_states[np.random.choice(
                start_states.shape[0],
                self.rand_per_mpc_step,
                replace=self.rand_per_mpc_step > start_states.shape[0]
            )]

        # run the explorer from those states
        explore_len = ((self.max_path_length + 1) * self.mpc_explore_len) - 1
        self.policy_ex_algo.max_path_length = explore_len
        ex_trajs = self.sampler_ex.obtain_samples(
            start_states.shape[0] * explore_len,
            explore_len, None, reset_args=start_states)

        # Now concat actions taken by explorer with observations for adding to the dataset
        trajs = ex_trajs['obs']
        obs = trajs[:, -1]
        if hasattr(self.action_space, 'shape') and len(self.action_space.shape) > 0:
            acts = get_numpy(ex_trajs['actions'])
        else:
            # convert discrete actions into onehot
            act_idx = get_numpy(ex_trajs['actions'])
            acts = np.zeros((trajs.shape[0], trajs.shape[1]-1, dataset.action_dim))
            acts_reshape = acts.reshape((-1, dataset.action_dim))
            acts_reshape[range(acts_reshape.shape[0]), act_idx.reshape(-1)] = 1.0

        # concat actions with obs
        acts = np.concatenate((acts, acts[:, -1:, :]), 1)
        trajacts = np.concatenate((ex_trajs['obs'], acts), axis=-1)
        trajacts = trajacts.reshape((-1, self.max_path_length + 1, trajacts.shape[-1]))
        
        # compute train/val split
        ntrain = min(int(0.9*trajacts.shape[0]), dataset.buffer_size//self.add_frac)
        if dataset.n < dataset.batch_size and ntrain < dataset.batch_size:
            ntrain = dataset.batch_size
        nvalid = min(trajacts.shape[0] - ntrain, test_dataset.buffer_size//self.add_frac)
        if test_dataset.n < test_dataset.batch_size and nvalid < test_dataset.batch_size: 
            nvalid = test_dataset.batch_size

        print("Adding ", ntrain, ", Valid: ", nvalid)

        dataset.add_samples(trajacts[:ntrain].reshape((ntrain, -1)))
        test_dataset.add_samples(trajacts[-nvalid:].reshape((nvalid, -1)))

        # dummy dataset stores only data from this iteration
        dummy_dataset.clear()
        dummy_dataset.add_samples(trajacts[:-nvalid].reshape((trajacts.shape[0]-nvalid, -1)))

        # compute negative ELBO on trajectories of explorer
        neg_elbos = []
        cur_batch = from_numpy(trajacts).float()
        for i in range(0, trajacts.shape[0], self.batch_size):
            mse, neg_ll, kl, bcloss, z_dist = self.vae.forward_batch(cur_batch[i:i+self.batch_size])
            neg_elbo = (get_numpy(neg_ll) + get_numpy(kl))
            neg_elbos.append(neg_elbo)

        # reward the explorer
        rewards = np.zeros_like(ex_trajs['rewards'])
        neg_elbos = np.concatenate(neg_elbos, axis=0)
        neg_elbos = neg_elbos.reshape((rewards.shape[0], -1))
        # just not on the first iteration, since VAE hasnt fitted yet
        if itr != 1:
            rewidx = list(range(self.max_path_length, explore_len, self.max_path_length + 1)) + [explore_len-1]
            for i in range(rewards.shape[0]):
                rewards[i, rewidx] = neg_elbos[i]

            # add in true reward to explorer if desired
            if self.true_reward_scale != 0:
                rstate = init_rstate(rewards.shape[0])
                for oidx in range(rewards.shape[1]):
                    r, rstate = reward_fn(ex_trajs['obs'][:, oidx], rstate) 
                    rewards[:, oidx] += r* self.true_reward_scale

        ex_trajs['rewards'] = rewards

        # train explorer using PPO with neg elbo 
        self.policy_ex_algo.process_samples(0, ex_trajs)#, augment_obs=get_numpy(z))
        if itr != 1:
            self.policy_ex_algo.optimize_policy(0, ex_trajs)
        ex_trajs['stats']['MPC Actual'] = np.mean(total_mpc_rew)
        ex_trajs['stats']['Final Reward'] = np.mean(final_reward)

        # reset explorer if necessary
        if ex_trajs['stats']['Entropy'] < self.reset_ent:
            if hasattr(self.policy_ex, "prob_network"):
                self.policy_ex.prob_network.apply(xavier_init)
            else:
                self.policy_ex.apply(xavier_init)
                self.policy_ex.log_var_network.params_var.data = self.policy_ex.log_var_network.param_init

        # for visualization purposes
        colors = ['purple', 'magenta', 'green', 'black', 'yellow', 'black']
        fig, ax = plt.subplots(3, 2, figsize=(10, 10))
        for i in range(6):
            if i*2+1 < obs.shape[1]:
                axx = ax[i//2][i%2]
                if i == 5:
                    axx.scatter(obs[:, -3], obs[:, -2], color=colors[i], s=10)
                else:
                    axx.scatter(obs[:, i*2], obs[:, i*2+1], color=colors[i], s=10)
                axx.set_xlim(-3, 3)
                axx.set_ylim(-3, 3)
        path = logger.get_snapshot_dir() + '/final_dist'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig('%s/%d.png' % (path, itr))
        np.save(path + "/" + str(itr), obs)
        
        return ex_trajs['stats']


    def sample_pd(self, sampler, latent, trajs):
        num_traj = latent.size()[0]
        print('sampling', num_traj , self.max_path_length)
        sd = sampler.obtain_samples(num_traj * self.max_path_length, self.max_path_length, latent, reset_args=get_numpy(trajs))
        return sd

    def forward(self, sampler, trajs, latent):
        pd_traj = self.sample_pd(sampler, latent, trajs)
        sd_traj = self.vae.decode(trajs, latent)

        return pd_traj, sd_traj

    def compute_traj_mse(self, traj1, traj2, traj_shape):
        return torch.pow(traj1.view(traj_shape) - traj2.view(traj_shape), 2).mean(-1)

    def plot_compare(self, dataset, itr):
        trajs, _ = dataset.sample_hard(self.plot_size)
        x, actdata = self.vae.splitobs(np_to_var(trajs))
        latent_dist = self.vae.encode(x)
        latent = latent_dist.sample(deterministic=True)

        #import pdb; pdb.set_trace()
        pd_traj, sd_traj = self.forward(self.plot_sampler, FloatTensor(trajs), latent)

        # sample for plottin
        traj_sets = [get_numpy(x)[:, self.step_dim:], get_numpy(sd_traj.mle), pd_traj['obs'][:, 1:]]
        traj_sets = [x.reshape((self.plot_size, self.max_path_length, -1)) for x in traj_sets]
        traj_names = ['expert', 'sd', 'pd']
        plot_traj_sets([dataset.process(x) for x in traj_sets], traj_names, itr, env_id=dataset.env_id)

        #dataset.plot_pd_compare([x[0] for x in traj_sets], traj_names, itr)
        for traj_no in range(5):
            dataset.plot_pd_compare([x[traj_no, ...] for x in traj_sets], traj_names, itr,
                                    name='Full_State_%d' % traj_no, save_dir='pd_match_expert')
        self.zero_grad()

    def save(self, snapshot_dir, itr):
        import os
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        torch.save(self.encoder.state_dict_lst(), snapshot_dir + '/encoder_%d.pkl' % itr)
        torch.save(self.decoder.state_dict_lst(), snapshot_dir + '/decoder_%d.pkl' % itr)
        torch.save(self.policy.state_dict_lst(), snapshot_dir + '/policy_%d.pkl' % itr)
        torch.save(self.policy_ex.state_dict_lst(), snapshot_dir + '/policy_ex_%d.pkl' % itr)
        torch.save(self.vae.optimizer.state_dict(), snapshot_dir + '/vae_optimizer_%d.pkl' % itr)
        torch.save(self.policy_algo.optimizer.state_dict(), snapshot_dir + '/policy_optimizer_%d.pkl' % itr)
        torch.save(self.policy_ex_algo.optimizer.state_dict(), snapshot_dir + '/policy_ex_optimizer_%d.pkl' % itr)

    def zero_grad(self):
        self.policy.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    # train pd to match sd using log likelihood of sd
    def train_pd_match_sd(self, dataset, bs, itr, outer_itr):
        sampler = self.sampler
        expert_traj, _ = dataset.sample(bs)

        # sample from dataset to initialize trajectory from
        x, actdata = self.vae.splitobs(FloatTensor(expert_traj))

        z = Variable(torch.randn((bs, self.latent_dim)))
        pd_traj, sd_traj = self.forward(sampler, x, z)
        sd_traj_obs = get_numpy(sd_traj.mle)

        traj_3d_shape = (bs, -1, self.obs_dim)

        pd_traj_obs = np_to_var(pd_traj['obs'][:, 1:])


        se = sd_traj.reshape(traj_3d_shape).log_likelihood(pd_traj_obs)
        mse_sd_pd = self.compute_traj_mse(pd_traj_obs, sd_traj.mle, traj_3d_shape)

        pd_traj['rewards'] = get_numpy(se)

        self.policy_algo.process_samples(0, pd_traj, augment_obs=get_numpy(z))
        self.policy_algo.optimize_policy(0, pd_traj)

        traj_sets = [sd_traj_obs, pd_traj['obs'][:, 1:]]
      
        pd_traj['stats']['mse_sd_pd'] = get_numpy(mse_sd_pd.mean())[0]
        pd_traj['stats']['ll'] = np.mean(get_numpy(se))

        return pd_traj['stats']

    def eval_rewards(self, obs, reward_fn, rstate, discount=False):
        rewards = np.zeros(obs.shape[0])
        for oidx in range(obs.shape[1]):
            r, rstate = reward_fn(obs[:, oidx], rstate)
            if discount:
                r *= self.discount_factor ** oidx
            rewards += r
        return rewards, rstate

    # batched rollout of sd
    def rollout_meta(self, latents, cur_obs, reward_fn, rstate):
        nbatch = latents.shape[1]
        state = cur_obs #np.array([cur_obs] * nbatch)
        trajs = []
        for lat in latents:
            latent_v = np_to_var(lat)
            state_v = from_numpy(state).float()
            sd_traj = self.vae.decode(state_v, latent_v)
            self.vae.decoder.zero_grad()
            decoded_traj = get_numpy(sd_traj.mle).reshape((nbatch, -1, cur_obs.shape[1]))
            state = decoded_traj[:, -1]
            trajs.append(decoded_traj)
        combo_traj = np.concatenate(trajs, axis=1)
        rewards, rstate = self.eval_rewards(combo_traj, reward_fn, rstate, discount=True)
        return rewards, combo_traj


    # batched mpc controller
    def mpc(self, cur_obs, horizon, batch_size, mpc_batch_size, reward_fn, rstate):
        def mpc_helper(obs, rstates):
            size = len(obs)
            bestlatents = np.zeros((size, self.latent_dim))
            latents = np.random.randn(horizon, size * batch_size, self.latent_dim)
            start_obs = np.concatenate([[o] * batch_size for o in obs], axis=0)
            start_rstate = np.concatenate([[r] * batch_size for r in rstates], axis=0)
            rtot, combo_traj = self.rollout_meta(latents, start_obs, reward_fn, start_rstate)
            rtot = rtot.reshape((size, batch_size))
            combo_traj = combo_traj.reshape((size, batch_size) + combo_traj.shape[1:])
            latents = latents.reshape(horizon, size, batch_size, self.latent_dim)
            amax = np.argmax(rtot, axis=1)
            bestlatents = np.swapaxes(latents[:, range(size), amax], 0,1)
            return np.max(rtot, axis=1), combo_traj[range(size), amax], bestlatents, combo_traj[0][:100]
        
        trajs = [[], [], [], []]
        n = len(cur_obs)
        for i in range(0, n, mpc_batch_size):
            batch = mpc_helper(cur_obs[i: i + mpc_batch_size], rstate[i: i + mpc_batch_size])
            for i in range(len(trajs)):
                trajs[i].append(batch[i])
        return [np.concatenate(x, axis=0) for x in trajs]
        
    # train vae on a single dataset
    def train_vae(self, dataset, test_dataset, outer_itr, itr):
        stats = self.vae.train_epoch(dataset, itr, max_steps=self.vae_train_steps)
        all_stats = {}
        stat_list = [(' T', stats)]
        stat_list.append((' TPD', self.train_pd_match_sd(dataset, 20, outer_itr, outer_itr)))

        if test_dataset is not None:
            test = self.vae.train_epoch(test_dataset, itr, train=False, max_steps=self.vae_train_steps//10)
            stat_list.append((' V', test))

        for prefix, stat in stat_list:
            for k, v in stat.items():
                all_stats[k + prefix] = v
        return all_stats

    # jointly train vae on two datasets
    def train_vae_joint(self, dataset, other_dataset, test_dataset, outer_itr, itr):

        all_stats = {}
        stat_list = [(' T', defaultdict(list)), (' N', defaultdict(list))]
        data_gen = [self.vae.loss_generator(dataset)]#, self.vae.loss_generator(other_dataset)]

        for i in range(self.vae_train_steps):
            losses = []
            self.vae.optimizer.zero_grad()
            for (_, stats), gen in zip(stat_list, data_gen):
                loss, stat_var = next(gen, (None, None))
                if loss is not None:
                    losses.append(loss)
                    for k, v in stat_var.items():
                        stats[k].append(get_numpy(v)[0])
            if len(losses) > 0:
                total_loss = sum(losses) / len(losses)
                total_loss.backward()
                self.vae.optimizer.step()
        for _, stats in stat_list:
            for k, v in stats.items():
                stats[k] = np.mean(v)

        if test_dataset is not None:
            test = self.vae.train_epoch(test_dataset, itr, train=False, max_steps=self.vae_train_steps//10)
            stat_list.append((' V', test))
        stat_list.append((' PD', self.train_pd_match_sd(dataset, 20, outer_itr, outer_itr)))

        for prefix, stat in stat_list:
            for k, v in stat.items():
                all_stats[k + prefix] = v
        return all_stats

