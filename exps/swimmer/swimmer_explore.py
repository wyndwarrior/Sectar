import argparse
from os import getcwd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traj2vec.config as config
from doodad.easy_sweep.hyper_sweep import Sweeper
from sklearn.model_selection import train_test_split
import traj2vec.launchers.config as launcher_config
from traj2vec.algos.ppo import PPO
from traj2vec.algos.vae_bc import TrajVAEBC
from traj2vec.algos.vaepdentropy import VAEPDEntropy
from traj2vec.algos.vpg import VPG
from traj2vec.datasets.dataset import WheeledContDataset
from traj2vec.envs.env_utils import make_env
from traj2vec.models.baselines.baseline import ZeroBaseline
from traj2vec.models.baselines.linear_feature_baseline import LinearFeatureBaseline
from traj2vec.models.baselines.nn_baseline import NNBaseline
from traj2vec.models.containers.categorical_network import RecurrentCategoricalPolicy, CategoricalNetwork, LSTMPolicy
from traj2vec.models.containers.gaussian_network import GaussianNetwork, GaussianRecurrentNetwork, \
    GaussianRecurrentPolicy, GaussianBidirectionalNetwork
from traj2vec.models.containers.mixed_network import MixedRecurrentNetwork
from traj2vec.nn.mlp import MLP
from traj2vec.nn.parameter import Parameter
from traj2vec.nn.rnn import RNN
from traj2vec.launchers.launcher_util_andrew import run_experiment
from traj2vec.nn.running_stat import ObsNorm
from traj2vec.utils.torch_utils import set_gpu_mode
from traj2vec.envs.swimmer import SwimmerEnv, reward_fn, init_rstate
import os
import sys
import json



def run_task(vv):
    set_gpu_mode(vv['gpu'])
    env_name = None

    goals = np.array(vv['goals'])
    env = lambda : SwimmerEnv(vv['frame_skip'], goals=goals, include_rstate=False) 

    obs_dim = int(env().observation_space.shape[0])
    action_dim = int(env().action_space.shape[0])
    vv['block_config'] = [env().reset().tolist(), vv['goals']]
    print(vv['block_config'])

    path_len = vv['path_len']
    data_path = vv['initial_data_path']
    use_actions = vv['use_actions']

    dummy = np.zeros((1, path_len+1, obs_dim + action_dim))
    train_data, test_data = dummy, dummy
    train_dataset = WheeledContDataset(data_path=data_path, raw_data=train_data, obs_dim=obs_dim, action_dim=action_dim, path_len=path_len,
                          env_id='Playpen', normalize=False, use_actions=use_actions, batch_size=vv['batch_size'],
                                      buffer_size=vv['buffer_size'], pltidx=[-2, -1])

    test_dataset = WheeledContDataset(data_path=data_path, raw_data=train_data, obs_dim=obs_dim, action_dim=action_dim, path_len=path_len,
                          env_id='Playpen', normalize=False, use_actions=use_actions, batch_size=vv['batch_size']//9,
                                      buffer_size=vv['buffer_size']//9, pltidx=[-2, -1])
    dummy_dataset = WheeledContDataset(data_path=data_path, raw_data=train_data, obs_dim=obs_dim, action_dim=action_dim, path_len=path_len,
                          env_id='Playpen', normalize=False, use_actions=use_actions, batch_size=vv['batch_size'],
                                      buffer_size=vv['buffer_size'], pltidx=[-2, -1])

    train_dataset.clear()
    test_dataset.clear()
    dummy_dataset.clear()

    latent_dim = vv['latent_dim']
    policy_rnn_hidden_dim = vv['policy_rnn_hidden_dim']
    rnn_hidden_dim = vv['decoder_rnn_hidden_dim']

    step_dim = obs_dim
    rnn_hidden_dim = 256
    if vv['encoder_type'] == 'mlp':
        encoder = GaussianNetwork(
            mean_network=MLP((path_len+1)*step_dim, latent_dim, hidden_sizes=vv['encoder_hidden_sizes'], hidden_act=nn.ReLU),
            log_var_network=MLP((path_len+1)*step_dim, latent_dim)
        )
    elif  vv['encoder_type'] == 'lstm':
        encoder = GaussianBidirectionalNetwork(
            input_dim=step_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=2,
            mean_network=MLP(2 * rnn_hidden_dim, latent_dim),
            log_var_network=MLP(2 * rnn_hidden_dim, latent_dim)
        )

    if vv['decoder_var_type'] == 'param':
        decoder_log_var_network = Parameter(latent_dim, step_dim, init=np.log(0.1))
    else:
        decoder_log_var_network = MLP(rnn_hidden_dim, step_dim)
    if vv['decoder_type'] == 'grnn':
        decoder = GaussianRecurrentNetwork(
            recurrent_network=RNN(nn.LSTM(step_dim + latent_dim, rnn_hidden_dim), rnn_hidden_dim),
            mean_network=MLP(rnn_hidden_dim, step_dim, hidden_sizes=vv['decoder_hidden_sizes'], hidden_act=nn.ReLU),
            log_var_network=decoder_log_var_network,
            path_len=path_len,
            output_dim=step_dim,
            min_var=1e-4,
        )
    elif vv['decoder_type'] == 'gmlp':
        decoder = GaussianNetwork(
            mean_network=MLP(latent_dim, path_len*step_dim, hidden_sizes=vv['decoder_hidden_sizes'],
                             hidden_act=nn.ReLU),
            log_var_network=Parameter(latent_dim, path_len*step_dim, init=np.log(0.1)),
            min_var=1e-4
        )
    elif vv['decoder_type'] == 'mixedrnn':
        gauss_output_dim = 10
        cat_output_dim = 5
        decoder = MixedRecurrentNetwork(
            recurrent_network=RNN(nn.LSTM(step_dim + latent_dim, rnn_hidden_dim), rnn_hidden_dim),
            mean_network=MLP(rnn_hidden_dim, gauss_output_dim, hidden_sizes=vv['decoder_hidden_sizes'], hidden_act=nn.ReLU),
            prob_network=MLP(rnn_hidden_dim, cat_output_dim, final_act=nn.Softmax),
            log_var_network=Parameter(latent_dim, gauss_output_dim, init=np.log(0.1)),
            path_len=path_len,
            output_dim=step_dim,
            min_var=1e-4,
            gaussian_output_dim=gauss_output_dim,
            cat_output_dim=cat_output_dim
        )

    if vv['policy_type'] == 'grnn':
        policy = GaussianRecurrentPolicy(
            recurrent_network=RNN(nn.LSTM(obs_dim+latent_dim, policy_rnn_hidden_dim), policy_rnn_hidden_dim),
            mean_network=MLP(policy_rnn_hidden_dim, action_dim,
                             hidden_act=nn.ReLU),
            log_var_network=Parameter(obs_dim+latent_dim, action_dim, init=np.log(1)),
            path_len=path_len,
            output_dim=action_dim
        )

    elif vv['policy_type'] == 'gmlp':
        policy = GaussianNetwork(
            mean_network=MLP(obs_dim+latent_dim, action_dim, hidden_sizes=vv['policy_hidden_sizes'],
                             hidden_act=nn.ReLU),
            log_var_network=Parameter(obs_dim+latent_dim, action_dim, init=np.log(1))
        )
        policy_ex = GaussianNetwork(
            mean_network=MLP(obs_dim, action_dim, hidden_sizes=vv['policy_hidden_sizes'],
                             hidden_act=nn.ReLU),
            log_var_network=Parameter(obs_dim, action_dim, init=np.log(1))
        )
    elif vv['policy_type'] == 'crnn':
        policy = RecurrentCategoricalPolicy(
            recurrent_network=RNN(nn.LSTM(obs_dim + latent_dim, policy_rnn_hidden_dim), policy_rnn_hidden_dim),
            prob_network=MLP(policy_rnn_hidden_dim, action_dim, hidden_sizes=vv['policy_hidden_sizes'], final_act=nn.Softmax),
            path_len=path_len,
            output_dim=action_dim
        )
    elif vv['policy_type'] == 'cmlp':
        policy = CategoricalNetwork(
            prob_network=MLP(obs_dim+ latent_dim, action_dim, hidden_sizes=(400, 300, 200),
                             hidden_act=nn.ReLU, final_act=nn.Softmax),
            output_dim=action_dim
        )
        policy_ex = CategoricalNetwork(
            prob_network=MLP(obs_dim, action_dim, hidden_sizes=(400, 300, 200),
                             hidden_act=nn.ReLU, final_act=nn.Softmax),
            output_dim=action_dim
        )
    elif vv['policy_type'] == 'lstm':
        policy = LSTMPolicy(input_dim = obs_dim + latent_dim,
                     hidden_dim = rnn_hidden_dim, 
                     num_layers = 2, 
                     output_dim = action_dim)

    vae = TrajVAEBC(encoder=encoder, decoder=decoder, latent_dim=latent_dim, step_dim=step_dim,
                  feature_dim=train_dataset.obs_dim, env=env, path_len=train_dataset.path_len,
                  init_kl_weight=vv['kl_weight'], max_kl_weight=vv['kl_weight'], kl_mul=1.03,
                  loss_type=vv['vae_loss_type'], lr=vv['vae_lr'], obs_dim=obs_dim,
                  act_dim=action_dim, policy=policy, bc_weight=vv['bc_weight'])

    baseline = ZeroBaseline()
    policy_algo = PPO(env, env_name, policy, baseline=baseline, obs_dim=obs_dim,
                             action_dim=action_dim, max_path_length=path_len, center_adv=True,
                     optimizer=optim.Adam(policy.get_params(), vv['policy_lr'], eps=1e-5), 
                      use_gae=vv['use_gae'], epoch=10, ppo_batch_size=200)

    baseline_ex = ZeroBaseline()
    policy_ex_algo = PPO(env, env_name, policy_ex, baseline=baseline_ex, obs_dim=obs_dim,
                             action_dim=action_dim, max_path_length=path_len, center_adv=True,
                     optimizer=optim.Adam(policy_ex.get_params(), vv['policy_lr'], eps=1e-5), 
                      use_gae=vv['use_gae'], epoch=10, ppo_batch_size=200,
                      entropy_bonus = vv['entropy_bonus'])


    if vv['load_models_dir'] is not None:
        dir = getcwd() + "/research/lang/traj2vecv3_jd/" + vv['load_models_dir']
        itr = vv['load_models_idx']
        encoder.load_state_dict(torch.load(dir + '/encoder_%d.pkl' % itr))
        decoder.load_state_dict(torch.load(dir + '/decoder_%d.pkl' % itr))
        policy.load_state_dict(torch.load(dir + '/policy_%d.pkl' % itr))
        policy_ex.load_state_dict(torch.load(dir + '/policy_ex_%d.pkl' % itr))
        vae.optimizer.load_state_dict(torch.load(dir + '/vae_optimizer_%d.pkl' % itr))
        policy_algo.optimizer.load_state_dict(torch.load(dir + '/policy_optimizer_%d.pkl' % itr))


    rf = lambda obs, rstate: reward_fn(obs, rstate, goals, 3)
    mpc_explore = 4000
    if vv['path_len'] <= 50:
        mpc_explore *= 2
    vaepd = VAEPDEntropy(env, env_name, policy, policy_ex, encoder, decoder,
        path_len, obs_dim, action_dim, step_dim, policy_algo, policy_ex_algo,
                  train_dataset, latent_dim, vae,
                  batch_size=400,
                  block_config=vv['block_config'],
                  plan_horizon = vv['mpc_plan'], 
                  max_horizon = vv['mpc_max'], 
                  mpc_batch = vv['mpc_batch'],
                  rand_per_mpc_step = vv['mpc_explore_step'],
                  mpc_explore = mpc_explore, 
                  mpc_explore_batch = 1,
                  reset_ent = vv['reset_ent'],
                  vae_train_steps = vv['vae_train_steps'],
                  mpc_explore_len=vv['mpc_explore_len'],
                  true_reward_scale=vv['true_reward_scale'],
                  discount_factor=vv['discount_factor'],
                  reward_fn=(rf, init_rstate)
                  )

    vaepd.train(train_dataset, test_dataset=test_dataset, dummy_dataset=dummy_dataset, plot_step=10, max_itr=vv['max_itr'], record_stats=True, print_step=1000,
                             save_step=2, 
               start_itr=0, train_vae_after_add=vv['train_vae_after_add'],
                joint_training=vv['joint_training'])



parser = argparse.ArgumentParser()

variant_group = parser.add_argument_group('variant')
variant_group.add_argument('--algo', default='entropy')
variant_group.add_argument('--debug', default='None')
variant_group.add_argument('--gpu', default="True")
variant_group.add_argument('--exp_dir', default='tmp')
variant_group.add_argument('--mode', default='local')
variant_group.add_argument('--load_models_dir', default=None)
variant_group.add_argument('--load_models_idx', default=None, type=int)
variant_group.add_argument('--env_name', default='SwimmerEnv')
variant_group.add_argument('--max_itr', default=1000, type=int)
variant_group.add_argument('--initial_data_path', default='/../traj2vecv3_master/data/test_data/playpen/block_bc.npz')

v_command_args = parser.parse_args()
command_args = {k.dest:vars(v_command_args)[k.dest] for k in variant_group._group_actions}
launcher_config.DIR_AND_MOUNT_POINT_MAPPINGS.append(
    dict(local_dir=getcwd() + command_args['initial_data_path'],
         mount_point='/root/code' + command_args['initial_data_path']))


goals = np.load('goals/swim.npy').tolist()

params = {
    'path_len': [99],
    'goals':goals,
    'mpc_plan': [30],
    'mpc_max': [50], 
    'frame_skip': [200],
    'add_frac': [100],
    'vae_train_steps': [30],
    'reset_ent': [0],
    'mpc_batch': [20],
    'mpc_explore_step':[200],
    'mpc_explore_len':[2],
    'sparse_reward':[True],
    'joint_training':[False],
    'consis_finetuning':[False],
    'true_reward_scale':[0],
    'discount_factor':[0.99],
    'policy_type': ['gmlp'],
    'policy_rnn_hidden_dim': [128],
    'policy_hidden_sizes': [(400, 300, 200)],
    'random_action_p': [0],
    # Encoder / Decoder
    'encoder_type': ['lstm'],
    'latent_dim': [8],
    'use_actions': [True],
    'encoder_hidden_sizes': [(128, 128, 128)],
    'decoder_hidden_sizes': [(128, 128, 128)],
    'decoder_rnn_hidden_dim': [512],
    'decoder_type': ['grnn'],
    'decoder_var_type': ['param'],
    # Buffer
    'initial_data_size': [9000],
    'buffer_size': [1000000],
    
    'vae_loss_type': ['ll'],
    'kl_weight': [2],
    
    'vae_lr': [1e-3],
    'policy_lr': [3e-4],
    'entropy_bonus': [1e-3],
    'use_gae': [True],
    'train_vae_after_add': [10],
    'batch_size': [300],
    'seed': [111],
    'bc_weight' : [100],
}

exp_id = 0
command_args['gpu'] = command_args['gpu'] == 'True'
for args in Sweeper(params, 1):

    env_name = command_args['env_name'].split('-')[0]
    alg_name = command_args['algo']
    exp_dir = command_args['exp_dir']
    print("gpu", command_args['gpu'], type(command_args['gpu']))
    if command_args['debug'] != 'None':
        with open(command_args['debug'] + "variant.json", 'r') as f:
            args = json.loads(f.read())
        for k in ('exp_id','seed'):
            args[k] = int(args[k])
        command_args['load_models_dir'] = command_args['debug'] + "snapshots/"
    base_log_dir = getcwd() + '/data/%s/%s/%s' % (alg_name, env_name, exp_dir)

    run_experiment(
        run_task,
        exp_id=exp_id,
        use_gpu=command_args['gpu'],
        mode=command_args['mode'],
        seed=args['seed'],
        prepend_date_to_exp_prefix=False,
        exp_prefix='%s-%s-%s' %(env_name, alg_name, exp_dir),
        base_log_dir=base_log_dir,
        variant={**args, **command_args},
    )
    if command_args['debug'] != 'None':
        sys.exit(0)
    exp_id += 1
