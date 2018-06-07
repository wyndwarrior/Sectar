import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

import sandbox.vime.algos.trpo_expl
from sandbox.vime.algos.trpo_expl import TRPO as TRPOVIME
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

from gym.envs.registration import register
import numpy as np
from doodad.easy_sweep.hyper_sweep import Sweeper
from os import getcwd
from traj2vec.launchers.launcher_util_andrew import run_experiment
import rllab.misc.logger as rllablogger
import traj2vec.utils.logger as trajlogger
import os.path as osp
from rllab.envs.gym_env import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import argparse

from rllab.algos.trpo import TRPO

from mpc.main_solution import train_dagger
from traj2vec.envs.swimmer import SwimmerEnv, reward_fn, init_rstate
import tensorflow as tf
import sys
def setup_rllab_logging(vv):
    log_dir = trajlogger.get_snapshot_dir()
    tabular_log_file = osp.join(log_dir, 'rllprogress.csv')
    text_log_file = osp.join(log_dir, 'rlldebug.log')
    rllablogger.add_text_output(text_log_file)
    rllablogger.add_tabular_output(tabular_log_file)
    rllablogger.set_snapshot_dir(log_dir)
    rllablogger.set_snapshot_mode("gap")
    rllablogger.set_snapshot_gap(10)
    rllablogger.set_log_tabular_only(False)
    rllablogger.push_prefix("[%s] " % vv['exp_dir'])
    return log_dir

def get_env(vv):
    return SwimmerEnv(vv['frame_skip'], goals=np.array(vv['block_config']), include_rstate=True)

def run_vime(vv):
    setup_rllab_logging(vv)
    seed = vv['seed']
    eta = 0.0001
    path_len = vv['path_len']
    mdp = get_env(vv)
    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(300, 200, 100),
        init_std=1.0,
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = path_len * 100
    algo = TRPOVIME(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=path_len,
        n_itr=1000,
        step_size=0.01,
        eta=eta,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_replay_pool=True,
        use_kl_ratio=True,
        use_kl_ratio_q=True,
        n_itr_update=1,
        kl_batch_size=1,
        normalize_reward=False,
        replay_pool_size=1000000,
        n_updates_per_sample=5000,
        second_order_update=True,
        unn_n_hidden=[32],
        unn_layers_type=[1, 1],
        unn_learning_rate=0.0001
    )
    algo.train()

def run_trpo(vv):
    setup_rllab_logging(vv)
    path_len = vv['path_len']
    env = get_env(vv)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(300, 200, 100),
        init_std=1.0,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100*path_len,
        max_path_length=path_len,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()


def run_mpc(vv):
    log_dir = setup_rllab_logging(vv)
    env = get_env(vv)
    path_len = vv['path_len']
    goals =np.array(vv['block_config'])
    train_dagger(env=env,
                 cost_fn=lambda state, rstate: reward_fn(state, rstate, goals, 3),
                 logdir=log_dir,
                 init_rstate=init_rstate,
                 render=False,
                 learning_rate=1e-3,
                 dagger_iters=1000,
                 dynamics_iters=60,
                 batch_size=512,
                 num_paths_random=100,
                 num_paths_dagger=20,
                 num_simulated_paths=4000,
                 env_horizon=path_len,
                 mpc_horizon=99 * 30,
                 n_layers = 3,
                 size=200,
                 activation=tf.nn.relu,
                 output_activation=None,
                 plan_every=99,
                 )

parser = argparse.ArgumentParser()

variant_group = parser.add_argument_group('variant')
variant_group.add_argument('--exp_dir', default='tmp')
variant_group.add_argument('--gpu', default="False")
variant_group.add_argument('--mode', default='local')
v_command_args = parser.parse_args()
command_args = {k.dest:vars(v_command_args)[k.dest] for k in variant_group._group_actions}

for goalfile in ['goals/swim.npy']:

    goals = np.load(goalfile).tolist()
    params = {
        'block_config': goals,
        'frame_skip': [200],
        'seed': [111],
        'path_len': [99 * 50],
        'method': ['mpc', 'trpo',  'vime']
    }

    exp_id = 0
    methods = {'vime':run_vime,
        'trpo':run_trpo,
        'mpc': run_mpc}

    command_args['gpu'] = command_args['gpu'] == 'True'
    for args in Sweeper(params, 1):
        exp_dir = command_args['exp_dir']
        base_log_dir = getcwd() + '/data/baselines/%s/' % (exp_dir)
        use_gpu = command_args['gpu']
        instance_type = 'c4.2xlarge'
        if args['method'] == 'mpc':
            use_gpu = True
            instance_type = 'p2.xlarge'
        run_experiment(
            methods[args['method']],
            exp_id=exp_id,
            docker_image='dementrock/rllab3-shared',
            instance_type=instance_type,
            use_gpu=use_gpu,
            mode=command_args['mode'],
            seed=args['seed'],
            prepend_date_to_exp_prefix=False,
            exp_prefix=exp_dir,
            base_log_dir=base_log_dir,
            variant={**args, **command_args},
        )
        exp_id += 1






