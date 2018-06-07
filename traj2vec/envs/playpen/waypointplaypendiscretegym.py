
import numpy as np
from gym import spaces
from gym import utils
import gym.envs.mujoco 
# from rllab.core.serializable import Serializable
# from rllab import spaces
# from rllab.envs.mujoco.mujoco_env import MujocoEnv
from traj2vec.envs.env_utils import get_asset_xml

def reward_fn(obs, rstate, goals):
    goals_aug = np.concatenate((goals, np.zeros((1, 2))))
    ngoals = goals.shape[0]
    cur_goal = goals_aug[rstate]
    reward = np.zeros(obs.shape[0])
    good = (np.linalg.norm(obs[:, :2] - cur_goal, axis=-1) < 0.4) 
    reward[good & (rstate%3==2)] += 1 # 
    reward[rstate == ngoals] = 0
    rstate[good] += 1
    rstate = np.minimum(rstate, ngoals)
    return reward, rstate
def init_rstate(size):
    return np.zeros(size, dtype=int)

class WaypointPlayPenDiscreteGym(gym.envs.mujoco.MujocoEnv, utils.EzPickle):
    FILE = 'play_pen.xml'

    def __init__(
            self, goals=np.zeros((10, 2)), reset_args=None,
            include_rstate = False,
            nlp_sentence=["yellow", "cylinder", "blue"],
            *args, **kwargs):
        self.objects = ["cube", "sphere", "cylinder"]
        self.object_colors = ["red", "blue", "black", "yellow"]
        self.target_colors = ["orange", "black", "purple", "blue"]
        self.rstate = init_rstate(1)
        self.reset_args = reset_args
        self.include_rstate = include_rstate

        # TODO: Fill these in automatically
        # Dict of object to index in qpos
        self.object_dict = {(0, 0): [2, 3],
                            (1, 0): [4, 5],
                            (2, 1): [6, 7],
                            (3, 2): [8, 9]}
        self.goals = goals
        # Dict of targets to index in geom_pos
        self.target_dict = {0: 8, 1: 6, 2: 7, 3: 5}
        self.reward_fn = self.make_reward(nlp_sentence)
        self.attached_object = (-1, -1)
        self.all_objects = [(-1, -1), (0, 0), (1, 0), (2, 1), (3, 2)]
        self.threshold = 0.4
        self.move_distance = 0.2
        self.viewer = None
        utils.EzPickle.__init__(self)
        gym.envs.mujoco.MujocoEnv.__init__(self, get_asset_xml('play_pen.xml'), 1)
        self.action_space = spaces.Discrete(4)
        #Serializable.quick_init(self, locals())


    def get_current_obs(self):
        onehot = np.zeros(5)
        onehot[self.attached_object[0]+1] = 1
        #onehot = self.attached_object
        obslist = [
            np.array(self.model.data.qpos.flat)[:2],
        ]
        if self.include_rstate:
            obslist.append(onehot)
            ohrstate = np.zeros(self.goals.shape[0] + 1)
            ohrstate[self.rstate[0]] = 1.0
            obslist.append(ohrstate)
        return np.concatenate(obslist)

    def compute_obj_dist(self, trajs):
        n = trajs.shape[0]
        start_pos = np.array([2.6, 0, 2.6, 1.0, 2.6, -1.0, 2.6, 2.0])
        final_dist = np.square(trajs[:, -1, 2:10] - start_pos).reshape((n, 4, 2)).mean(-1) # bs, num_obj
        return final_dist.mean(-1).mean(-1)

    def compute_obj_pickup(self, trajs):
        pickup_idx = trajs[:, :, 10:].argmax(-1)
        mean_pickups = []
        for i in range(5):
            n_pickup = ((pickup_idx == i).sum(1) / trajs.shape[1]).mean()
            mean_pickups.append(n_pickup)
        return np.sum(mean_pickups[1:])

    def compute_stats(self, trajs, traj_name):
        # Trajs is numpy (bs, path_len, obs_dim)
        # Compute mean final obj distance from start for each obj
        n = trajs.shape[0]
        start_pos = np.array([2.6, 0, 2.6, 1.0, 2.6, -1.0, 2.6, 2.0])
        final_dist = np.square(trajs[:, -1, 2:10] - start_pos).reshape((n, 4, 2)).mean(-1) # bs, num_obj
 
       # Compute num pickups
        pickup_idx = trajs[:, :, 10:].argmax(-1)
        mean_pickups = []
        for i in range(5):
            n_pickup = ((pickup_idx == i).sum(1) / trajs.shape[1]).mean()
            mean_pickups.append(n_pickup)

        stats = {'mean final obj dist':final_dist.mean(-1).mean(-1),
                     'mean pickups':np.sum(mean_pickups[1:])
                    }
        new_stats = {}
        for k, v in stats.items():
            new_stats['%s %s' %(traj_name, k)] = v
        return new_stats

    def reset(self, reset_args = None):
        if reset_args is not None:
            self.reset_args = reset_args
        return super().reset()

    def reset_model(self):
        # TODO: Better reset function?
        # print(self.reset_args)
        self.rstate = init_rstate(1)
        full_qpos = np.zeros((10, 1))
        full_qpos[2, 0] = 0.0
        full_qpos[3, 0] = 0.0

        full_qpos[4, 0] = 0.0
        full_qpos[5, 0] = 0.0

        full_qpos[6, 0] = 0.0
        full_qpos[7, 0] = 0.0

        full_qpos[8, 0] = 0.0
        full_qpos[9, 0] = 0.0
        full_qpos[:2, 0] = self.reset_args[:2]

        self.model.data.qpos = full_qpos
        self.model._compute_subtree()
        self.model.forward()
        self.attached_object = (-1, -1)
        # if len(self.reset_args) > 10:
        #     self.attached_object = self.all_objects[np.argmax(self.reset_args[10:15])]
        return self.get_current_obs()

    def make_reward(self, nlp_sentence):
        color_to_move = self.object_colors.index(nlp_sentence[0])
        object_to_move = self.objects.index(nlp_sentence[1])
        target_index = self.target_colors.index(nlp_sentence[2])

        def reward_fn(obs):
            obj_pos = np.array([self.model.data.qpos[i, 0] for i in self.object_dict[(color_to_move, object_to_move)]])
            target_pos = self.model.geom_pos[self.target_dict[target_index]][0:2]
            dist_objectfist = -np.linalg.norm((obs[0:2] - obj_pos))
            dist_objectgoal = -np.linalg.norm((obj_pos - target_pos))
            return dist_objectfist + dist_objectgoal

        return reward_fn

    def _step(self, action):
        if not isinstance(action, int) and action.size > 1:
            action = 0
        self.move(action)
        next_obs = self.get_current_obs()
        reward, self.rstate = reward_fn(next_obs[None], self.rstate, self.goals)
        # print(reward, self.rstate, self.goals)
        done = False
        return next_obs, reward[0], done, {}

    # TODO deal with collisions
    def move(self, action):
        current_fist_pos = self.model.data.qpos[0:2].flatten()
        # TODO: Need to put action limits on this
        if action == 0:
            next_fist_pos = current_fist_pos + np.array([self.move_distance, 0])

        elif action == 1:
            next_fist_pos = current_fist_pos + np.array([0, self.move_distance])

        elif action == 2:
            next_fist_pos = current_fist_pos + np.array([-self.move_distance, 0])

        elif action == 3:
            next_fist_pos = current_fist_pos + np.array([0, -self.move_distance])

        # TODO: Less hardcoded
        next_fist_pos = np.clip(next_fist_pos, -2.8, 2.8)
        # Moving the objects jointly
        if self.attached_object != (-1, -1):
            current_obj_pos = np.array([self.model.data.qpos[i, 0] for i in self.object_dict[self.attached_object]])
            current_obj_pos += (next_fist_pos - current_fist_pos)

        # Setting the final positions
        curr_qpos = self.model.data.qpos.copy()
        curr_qpos[0, 0] = next_fist_pos[0]
        curr_qpos[1, 0] = next_fist_pos[1]
        if self.attached_object != (-1, -1):
            for enum_n, i in enumerate(self.object_dict[self.attached_object]):
                curr_qpos[i, 0] = current_obj_pos[enum_n]
        self.model.data.qpos = curr_qpos
        self.model._compute_subtree()
        self.model.forward()

    #@property
    def get_action_space(self):
        return spaces.Discrete(4)
