import numpy as np
import tensorflow as tf
import gym
from mpc.dynamics_solution import NNDynamicsModel
from mpc.controllers_solution import MPCcontroller, RandomController
from mpc.cost_functions import cheetah_cost_fn, \
                           hopper_cost_fn, \
                           swimmer_cost_fn, \
                           trajectory_cost_fn, get_playpen_cost
import time
import mpc.logz as logz
import os
import copy
import matplotlib.pyplot as plt
import rllab.misc.logger as logger
import pickle
from tqdm import tqdm

# plt.ion()

#Remove: Maybe
def sample(env,
           controller,
           num_paths=1,
           horizon=1000,
           render=False,
           verbose=False,
           timer=False):
    if hasattr(env.action_space, "shape"):
        act_dim = env.action_space.shape[0]
    else:
        act_dim = env.action_space.n
    paths = []
    for i in tqdm(range(num_paths)):
        ob = env.reset()
        obs, acs, rews = [], [], []
        steps = 0
        start_time = time.time()
        while True:
            # print(len(obs))
            # if timer:
            #     import IPython
            #     IPython.embed()
            for ac in controller.get_action(ob):
                obs.append(ob)
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rews.append(rew)
                # print(env.goal_idx, env.goalpos, ob[:2])
                steps += 1
                # if timer:
                #    print(steps)
                if done or steps > horizon:
                    if verbose:
                        logger.log('Finished one rollout. Took %.2f seconds. Attained return %.3f.'%(time.time()-start_time, sum(rews)))
                        print('Finished one rollout. Took %.2f seconds. Attained return %.3f.'%(time.time()-start_time, sum(rews)))
                    break
            if done or steps > horizon:
                break
        npacs = np.array(acs[:-1])
        if len(npacs.shape) == 1:
            oh_acts = np.zeros((npacs.shape[0], act_dim))
            oh_acts[range(npacs.shape[0]), npacs] = 1
            npacs = oh_acts
        path = {"observations" : np.array(obs[:-1]),
                "next_observations" : np.array(obs[1:]),
                "actions" : npacs,
                "rewards" : np.array(rews[:-1]),
                "return" : np.sum(rews[:-1])}
        paths.append(path)
    return paths

def path_cost(cost_fn, path, rstate):
    return trajectory_cost_fn(cost_fn, path['observations'][:, None], path['actions'][:, None],
        path['next_observations'][:, None], rstate)[0]

def process_data(paths):
    return dict(
        observations=np.concatenate([path["observations"] for path in paths]),
        actions=np.concatenate([path["actions"] for path in paths]),
        next_observations=np.concatenate([path["next_observations"] for path in paths])
        )

def append_data(old_data, new_paths):
    new_data = process_data(new_paths)
    return dict(
        observations=np.concatenate([old_data["observations"], new_data["observations"]]),
        actions=np.concatenate([old_data["actions"], new_data["actions"]]),
        next_observations=np.concatenate([old_data["next_observations"], new_data["next_observations"]])
        )

#Remove
def compute_normalization(data, epsilon=1e-2):
    ob_mean = np.mean(data["observations"], axis=0)
    ob_std = np.std(data["observations"], axis=0)
    ob_std = np.maximum(ob_std, epsilon * np.ones_like(ob_std))
    delta_mean = np.mean(data["next_observations"] - data["observations"], axis=0)
    delta_std = np.std(data["next_observations"] - data["observations"], axis=0)
    delta_std = np.maximum(delta_std, epsilon * np.ones_like(delta_std))
    ac_mean = np.mean(data["actions"], axis=0)
    ac_std = np.std(data["actions"], axis=0)
    ac_std = np.maximum(ac_std, epsilon * np.ones_like(ac_std))
    # ob_std[-1] = delta_std[-1]= 1
    return ob_mean, ob_std, delta_mean, delta_std, ac_mean, ac_std

#Remove
def plot_comparison(env, dyn_model, horizon=100, ax=None):
    return
    init_state = env.reset()

    actions =  np.random.uniform(env.action_space.low,
                                 env.action_space.high,
                                 size=(horizon, env.action_space.shape[0]))
    curr_state = copy.copy(init_state)
    states_sim = []
    for i in range(horizon):
        curr_state = dyn_model.predict([curr_state], [actions[i]])[0]
        states_sim.append(curr_state)

    states_actual = []
    for i in range(horizon):
        ob, rew, done, _ = env.step(actions[i])
        states_actual.append(ob)

    states_sim = np.asarray(states_sim)
    states_actual = np.asarray(states_actual)
    obs_dim = states_sim.shape[1]
    n = int(np.ceil(np.sqrt(obs_dim)))
    if ax is None:
        _, ax = plt.subplots(n, n)
    for j in range(n):
        for k in range(n):
            i = j * n + k
            if i < obs_dim:
                ax[j,k].clear()
                ax[j,k].plot(range(horizon), states_sim[:,i], '-')
                ax[j,k].plot(range(horizon), states_actual[:,i], '.')

    plt.pause(0.05)
    return ax

    """
    for i in range(states_sim.shape[1]):
        plt.plot(range(100), states_sim[:,i])
        plt.plot(range(100), states_actual[:,i])
        plt.show()
    """

def train_dagger(env,
                 cost_fn,
                 init_rstate,
                 logdir=None,
                 render=False,
                 learning_rate=1e-3,
                 dagger_iters=10,
                 dynamics_iters=60,
                 batch_size=512,
                 num_paths_random=10,
                 num_paths_dagger=10,
                 num_simulated_paths=1000,
                 env_horizon=20*30,
                 mpc_horizon=60,
                 n_layers=2,
                 size=500,
                 activation=tf.nn.relu,
                 output_activation=None,
                 plan_every=1,
                 ):

    """

    Arguments:

    dagger_iters                Number of iterations of the DAGGER loop to run.

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the outer DAGGER loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated
    |                           by a random agent. We use these to train our
    |_                          initial dynamics model.

    num_paths_dagger            Number of paths to collect at each iteration of
    |_                          DAGGER, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments.

    """

    logz.configure_output_dir(logdir)

    #========================================================
    #
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.
    #

    #Remove: Maybe
    random_controller = RandomController(env)

    random_paths = sample(env=env,
                          controller=random_controller,
                          num_paths=num_paths_random,
                          horizon=env_horizon)
    print("Using a dataset of %d random paths to start."%len(random_paths))
    D = process_data(random_paths)
    idxs = np.arange(len(D['observations']))

    np.random.shuffle(idxs)
    n = int(0.9*len(D['observations']))
    D_train = dict(
            observations=D['observations'][idxs[:n]],
            actions=D['actions'][idxs[:n]],
            next_observations=D['next_observations'][idxs[:n]]
            )
    D_test = dict(
            observations=D['observations'][idxs[n:]],
            actions=D['actions'][idxs[n:]],
            next_observations=D['next_observations'][idxs[n:]]
            )


    #========================================================
    #
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network.
    #
    normalization = compute_normalization(D_train)

    #import IPython
    #IPython.embed()

    #========================================================
    #
    # Build dynamics model and MPC controllers.
    #
    dyn_model = NNDynamicsModel(env=env,
                                n_layers=n_layers,
                                size=size,
                                activation=activation,
                                output_activation=output_activation,
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate)

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths,
                                   plan_every = plan_every)


    # import IPython
    # IPython.embed()
    #========================================================
    #
    # Tensorflow session building.
    #
    # tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.__enter__()
    tf.global_variables_initializer().run()

    dyn_model.sess = sess

    #========================================================
    #
    # Run the DAGGER loop for training the dynamics model.
    #
    # Important Note!
    #       Earlier in the course, we talked about DAGGER
    #       as a tool for behavior cloning, where we used it
    #       to learn a /policy/.
    #
    #       Here, we are using it to learn a /dynamics model/.
    #       The key concept is that DAGGER just means
    #       [Data AGGregation].
    #
    #       In our training loop, we
    #           1. Take some steps of gradient descent to fit
    #              the dynamics model to our CURRENT DATA.
    #
    #           2. We then act in the environment according to
    #              the MPC policy to collect NEW ROLLOUTS.
    #              The MPC policy /uses/ the dynamics model to
    #              select actions, but the dynamics model itself
    #              does not do anything to select actions.
    #
    #           3. We then augment CURRENT DATA with the NEW ROLLOUTS.
    #              (That is, we aggregate data.)
    #
    #        We /hope/ that if the dynamics model is really good,
    #        and esepcially good on high-reward states, this loop
    #        of improving the dynamics model will lead to a better
    #        MPC policy.
    #

    #Remove
    start_time = time.time()
    ax = None
    all_paths = random_paths
    for itr in range(dagger_iters):
        losses = dyn_model.fit(D_train, D_test)

        # ax = plot_comparison(env, dyn_model, 100, ax)


        new_paths = sample(env=env,
                           controller=mpc_controller,
                           num_paths=num_paths_dagger,
                           horizon=env_horizon,
                           render=False,
                           verbose=True,
                           timer = True)
        all_paths += new_paths
        n_step_error = dyn_model.n_step_error(all_paths, n=mpc_horizon)

        with open(logger.get_snapshot_dir() + "/latestobs.pkl", 'wb') as pfile:
            pickle.dump(new_paths, pfile)

        costs = [path_cost(cost_fn, path, init_rstate(1)) for path in new_paths]
        returns = [path['return'] for path in new_paths]

        #if itr==0:
        #    D_rl = process_data(new_paths)
        #else:
        #    D_rl = append_data(D_rl, new_paths)
        D_train = append_data(D_train, new_paths)

        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('Time', time.time() - start_time)

        # In terms of cost function
        logger.record_tabular('AverageCost', np.mean(costs))
        logger.record_tabular('StdCost', np.std(costs))
        logger.record_tabular('MinimumCost', np.min(costs))
        logger.record_tabular('MaximumCost', np.max(costs))

        # In terms of true reward
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn', np.std(returns))
        logger.record_tabular('MinimumReturn', np.min(returns))
        logger.record_tabular('MaximumReturn', np.max(returns))


        # L2 loss function for dynamics learning
        logger.record_tabular('AverageDynamicsLoss', np.mean(losses))
        logger.record_tabular('MinimumDynamicsLoss', np.min(losses))
        logger.record_tabular('DynamicsTestLoss', dyn_model.prediction_error(D_test))
        logger.record_tabular('NStepDynamicsLoss', n_step_error)

        logger.dump_tabular()
