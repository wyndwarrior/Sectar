import os

import gym
# from baselines import bench
from gym.envs.registration import register

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')
_REGISTERED = False

def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)

def make_env(env_id, seed, rank, log_dir, kwargs=None, monitor=False):
    global _REGISTERED
    if not _REGISTERED:
        register(id='BlockPlaypen-v0', entry_point='traj2vec.envs.playpen.blockplaypengym:BlockPlayPenGym',
            kwargs=kwargs)
        register(id='WaypointPlaypenDiscrete-v0', entry_point='traj2vec.envs.playpen.waypointplaypendiscretegym:WaypointPlayPenDiscreteGym',
            kwargs=kwargs)
    _REGISTERED = True

    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)
        if monitor:
            # env = bench.Monitor(env,
            #                     os.path.join(log_dir,
            #                                  "{}.monitor.json".format(rank)),
            #                     allow_early_resets=True)
            env = gym.wrappers.Monitor(env, os.path.join(log_dir, 'monitor.json'), force=True)
        return env

    return _thunk