
import numpy as np
from gym import spaces
from gym import utils
import gym.envs.mujoco
from traj2vec.envs.env_utils import get_asset_xml


def init_rstate(size):
    return np.zeros(size, dtype=int)
def reward_fn(obs, rstate, goals):
    diff = obs[:, 2:10] - goals
    diff = diff.reshape((-1, 4, 2))
    dists = np.linalg.norm(diff, axis=-1)
    dists = np.concatenate((dists, np.zeros((dists.shape[0], 1))), axis=1)
    good = dists < 0.8
    good = good[range(dists.shape[0]), rstate]
    reward = np.zeros(obs.shape[0])
    reward[good] += 1
    reward[rstate == 4] = 0
    rstate[good] += 1
    rstate = np.minimum(rstate, 4)
    return reward, rstate

class BlockPlayPenGym(gym.envs.mujoco.MujocoEnv, utils.EzPickle):
    FILE = 'play_pen.xml'

    def __init__(
            self,
            nlp_sentence=["yellow", "cylinder", "blue"],
            reset_args=None,
            goals=np.zeros((8)),
            include_rstate = False,
            border=2.8,
            *args, **kwargs):
        self.objects = ["cube", "sphere", "cylinder"]
        self.object_colors = ["red", "blue", "black", "yellow"]
        self.target_colors = ["orange", "black", "purple", "blue"]
        self.border = border
        self.rstate = init_rstate(1)
        self.reset_args = reset_args
        self.goals = goals
        self.include_rstate = include_rstate
        self.object_dict = {(0, 0): [2, 3],
                            (1, 0): [4, 5],
                            (2, 1): [6, 7],
                            (3, 2): [8, 9]}
        # Dict of targets to index in geom_pos
        self.target_dict = {0: 8, 1: 6, 2: 7, 3: 5}
        self.attached_object = (-1, -1)
        self.all_objects = [(-1, -1), (0, 0), (1, 0), (2, 1), (3, 2)]
        self.threshold = 0.4
        self.move_distance = 0.2
        self.viewer = None
        self.reward_fn = None
        self.init_rstate = None
        utils.EzPickle.__init__(self)
        gym.envs.mujoco.MujocoEnv.__init__(self, get_asset_xml('play_pen.xml'), 1)
        self.action_space = spaces.Discrete(6)


    def get_current_obs(self):
        onehot = np.zeros(5)
        onehot[self.attached_object[0]+1] = 1
        obslist = [
            self.model.data.qpos.flat,
            onehot
        ]
        if self.include_rstate:
            ohrstate = np.zeros(5)
            ohrstate[self.rstate[0]] = 1.0
            obslist.append(ohrstate)
        return np.concatenate(obslist)

    def compute_obj_dist(self, trajs):
        n = trajs.shape[0]
        start_pos = np.array([2.6, 0, 2.6, 1.0, 2.6, -1.0, 2.6, 2.0])
        final_dist = np.square(trajs[:, -1, 2:10] - start_pos).reshape((n, 4, 2)).mean(-1) 
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
        full_qpos = np.zeros((10, 1))
        full_qpos[2, 0] = 2.6
        full_qpos[3, 0] = 0.0

        full_qpos[4, 0] = 2.6
        full_qpos[5, 0] = 1.0

        full_qpos[6, 0] = 2.6
        full_qpos[7, 0] = -1.0

        full_qpos[8, 0] = 2.6
        full_qpos[9, 0] = 2.0
        full_qpos[:10, 0] = self.reset_args[:10]
        if self.init_rstate is not None:
            self.rstate = self.init_rstate(1)

        self.model.data.qpos = full_qpos
        self.model._compute_subtree()
        self.model.forward()
        self.attached_object = (-1, -1)
        if len(self.reset_args) > 10:
            self.attached_object = self.all_objects[np.argmax(self.reset_args[10:15])]
        return self.get_current_obs()

    def _step(self, action):
        if not isinstance(action, int) and action.size > 1:
            action = 0
        self.move(action)
        next_obs = self.get_current_obs()
        reward, self.rstate = reward_fn(next_obs[None], self.rstate, self.goals)
        done = False
        return next_obs, reward[0], done, {}

    # TODO deal with collisions
    def move(self, action):
        current_fist_pos = self.model.data.qpos[0:2].flatten()
        # TODO: Need to put action limits on this
        if action < 4:
            if action == 0:
                next_fist_pos = current_fist_pos + np.array([self.move_distance, 0])

            elif action == 1:
                next_fist_pos = current_fist_pos + np.array([0, self.move_distance])

            elif action == 2:
                next_fist_pos = current_fist_pos + np.array([-self.move_distance, 0])

            elif action == 3:
                next_fist_pos = current_fist_pos + np.array([0, -self.move_distance])

            # TODO: Less hardcoded
            next_fist_pos = np.clip(next_fist_pos, -self.border, self.border)
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
        else:
            if action == 4:
                # GRASP
                if self.attached_object == (-1, -1):
                    current_fist_pos = self.model.data.qpos[0:2].flatten()
                    all_object_positions = []
                    for k, v in self.object_dict.items():
                        curr_obj_pos = np.array([self.model.data.qpos[i, 0] for i in v])
                        dist = np.linalg.norm((current_fist_pos - curr_obj_pos))
                        if dist < self.threshold:
                            self.attached_object = k
                            return
            elif action == 5:
                # Drop
                self.attached_object = (-1, -1)

    #@property
    def get_action_space(self):
        return spaces.Discrete(6)