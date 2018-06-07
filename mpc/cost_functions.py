import numpy as np


#========================================================
#
# Environment-specific cost functions:
#

def get_playpen_cost(goalpos):
    def playpen_cost_fn(state, action, next_state, goal_idx):
        # import IPython
        # IPython.embed()
        if len(state.shape) > 1:
            dist = -np.linalg.norm((state[:, 0:2] - goalpos[goal_idx]), axis=1)
            # import IPython
            # IPython.embed()
            rew = 0 * dist
            for i in range(state.shape[0]):
                if np.abs(dist[i]) < 0.4:
                    if goal_idx[i] < len(goalpos):
                        goal_idx[i] += 1
                        rew[i] += 100
            return -rew
        asdf
    return playpen_cost_fn

def cheetah_cost_fn(state, action, next_state):
    if len(state.shape) > 1:

        heading_penalty_factor=10
        scores=np.zeros((state.shape[0],))

        #dont move front shin back so far that you tilt forward
        front_leg = state[:,5]
        my_range = 0.2
        scores[front_leg>=my_range] += heading_penalty_factor

        front_shin = state[:,6]
        my_range = 0
        scores[front_shin>=my_range] += heading_penalty_factor

        front_foot = state[:,7]
        my_range = 0
        scores[front_foot>=my_range] += heading_penalty_factor

        scores-= (next_state[:,17] - state[:,17]) / 0.01 #+ 0.1 * (np.sum(action**2, axis=1))
        return scores

    heading_penalty_factor=10
    score = 0

    #dont move front shin back so far that you tilt forward
    front_leg = state[5]
    my_range = 0.2
    if front_leg>=my_range:
        score += heading_penalty_factor

    front_shin = state[6]
    my_range = 0
    if front_shin>=my_range:
        score += heading_penalty_factor

    front_foot = state[7]
    my_range = 0
    if front_foot>=my_range:
        score += heading_penalty_factor

    score -= (next_state[17] - state[17]) / 0.01 #+ 0.1 * (np.sum(action**2))
    return score

def hopper_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        return -(next_state[:, 11] - state[:, 11]) / 0.01
    return -(next_state[11] - state[11]) / 0.01

def swimmer_cost_fn(state, action, next_state):

    penalty_factor = 0.2

    if len(state.shape) > 1:

        #scores = 0
        #"""
        scores=np.zeros((state.shape[0],))

        #dont bend in too much
        first_joint = np.abs(state[:,3])
        second_joint = np.abs(state[:,4])
        limit = np.pi/3
        scores[limit<first_joint] += penalty_factor
        scores[limit<second_joint] += penalty_factor
        #"""

        return scores -(next_state[:,-2] - state[:,-2]) / 0.01

    first_joint = np.abs(state[3])
    second_joint = np.abs(state[4])
    limit = np.pi/3
    score = penalty_factor*((limit < first_joint) + (limit < second_joint))
    return score -(next_state[-2] - state[-2]) / 0.01

#========================================================
#
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states, rstate):
    trajectory_cost = 0
    for i in range(len(states)):
        r, rstate = cost_fn(states[i], rstate)
        trajectory_cost += r
    # print(trajectory_cost)
    return trajectory_cost
