import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from traj2vec.envs.env_utils import get_asset_xml


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

def reward_fn(obs, rstate, goals):
    goals_aug = np.concatenate((goals, np.zeros((1, 2))))
    ngoals = goals.shape[0]
    cur_goal = goals_aug[rstate]
    reward = np.zeros(obs.shape[0])
    good = (np.linalg.norm(obs[:, :2] - cur_goal, axis=-1) < 0.6)
    reward[good & (rstate%3==2)] += 1
    reward[rstate == ngoals] = 0
    rstate[good] += 1
    rstate = np.minimum(rstate, ngoals)
    return reward, rstate
def init_rstate(size):
    return np.zeros(size, dtype=int)


class WheeledEnv(MujocoEnv, Serializable):

    FILE = 'wheeledasdf.xml'

    def __init__(self, goals=np.zeros((10, 2)), include_rstate=False, *args, **kwargs):
        self.rstate = init_rstate(1)
        self.include_rstate = include_rstate
        self.goals = goals
        kwargs['file_path'] = get_asset_xml('wheeled.xml')
        super(WheeledEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.frame_skip = 3


    def get_current_obs(self):
        qpos = list(self.model.data.qpos.flat)
        obslist = [
            qpos[:-3],
            [np.sin(qpos[-3]), np.cos(qpos[-3])],
            list(self.model.data.qvel.flat)[:-2],
        ]
        if self.include_rstate:
            ohrstate = np.zeros(self.goals.shape[0] + 1)
            ohrstate[self.rstate[0]] = 1.0
            obslist.append(ohrstate)
        return np.concatenate(obslist)

    @overrides
    def reset(self, reset_args=None, init_state=None, **kwargs):
        self.rstate = init_rstate(1)
        radius = 2.0
        angle = np.random.uniform(0, np.pi)
        xpos = radius*np.cos(angle)
        ypos = radius*np.sin(angle)
        self.goal = np.array([xpos, ypos])
        if reset_args is None:
            self.reset_mujoco(None)
        else:
            rstate = np.zeros_like(self._full_state)
            numpos = len(list(self.model.data.qpos.flat))
            endidx2 = numpos-2
            endidx3 = numpos-3
            zangle = np.arctan2(reset_args[endidx3], reset_args[endidx3+1])
            reset_args = reset_args[:endidx3 + endidx2 + 2]
            rstate[:endidx3] = reset_args[:endidx3]
            rstate[endidx3] = zangle
            rstate[numpos:numpos+endidx2] = reset_args[endidx3+2:]
            self.reset_mujoco(rstate)

        body_pos = self.model.body_pos.copy()
        body_pos[-1][:2] = self.goal
        self.model.body_pos = body_pos
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        reward, self.rstate = reward_fn(next_obs[None], self.rstate, self.goals)
        done = False
        return next_obs, reward[0], done, {}