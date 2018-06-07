from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
from traj2vec.envs.env_utils import get_asset_xml

def reward_fn(obs, rstate, goals, every):
    goals_aug = np.concatenate((goals, np.zeros((1, 2))))
    ngoals = goals.shape[0]
    cur_goal = goals_aug[rstate]
    reward = np.zeros(obs.shape[0])
    good = (np.linalg.norm(obs[:, 6:8] - cur_goal, axis=-1) < 0.2)
    reward[good & (rstate%every==(every-1))] += 1
    reward[rstate == ngoals] = 0
    rstate[good] += 1
    rstate = np.minimum(rstate, ngoals)
    return reward, rstate
def init_rstate(size):
    return np.zeros(size, dtype=int)

class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer_velasdf.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,frame_skip, goals, include_rstate,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.goals = goals
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.include_rstate = include_rstate
        kwargs['file_path'] = get_asset_xml('swimmer_vel.xml')
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.frame_skip = frame_skip

    def get_current_obs(self):
        qpos = list(self.model.data.qpos.flat)
        obslist = [
            qpos[:2],
            qpos[3:],
            [np.sin(qpos[2]), np.cos(qpos[2])],
            list(self.get_body_com("torso").flat)[:2],
        ]
        if self.include_rstate:
            ohrstate = np.zeros(len(self.goals) + 1)
            ohrstate[self.rstate[0]] = 1.0
            obslist.append(ohrstate)
        return np.concatenate(obslist)


    @overrides
    def reset(self, reset_args=None, init_state=None, **kwargs):
        self.rstate = init_rstate(1)
        if reset_args is None:
            self.reset_mujoco(None)
        else:
            rstate = np.zeros_like(self._full_state)
            numpos = len(list(self.model.data.qpos.flat))
            zangle = np.arctan2(reset_args[numpos-1], reset_args[numpos])
            rstate[:2] = reset_args[:2]
            rstate[2] = zangle
            rstate[3:numpos] = reset_args[2: numpos - 1]
            self.reset_mujoco(rstate)

        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        
        return obs

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        lb, ub = self.action_space.bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action1 = np.clip(scaled_action, lb, ub)
        self.forward_dynamics(scaled_action1)
        next_obs = self.get_current_obs()
        forward_reward = self.get_body_comvel("torso")[0]
        reward, self.rstate = reward_fn(next_obs[None], self.rstate, self.goals, 3)
        done = False
        return Step(next_obs, reward[0], done)
