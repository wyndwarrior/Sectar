import numpy as np
from mpc.cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.action_space = env.action_space

	def get_action(self, state):
		return [self.action_space.sample()]


class MPCcontroller(Controller):
	def __init__(self, 
				 env, 
				 dyn_model, 
				 plan_every,
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,

				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.obs_dim = env.observation_space.shape[0]
		self.plan_every = plan_every
		if hasattr(env.action_space, "shape"):
			self.act_dim = env.action_space.shape[0]
			self.lb = env.action_space.low
			self.ub = env.action_space.high
			self.discrete = False
		else:
			self.act_dim = env.action_space.n
			self.discrete = True
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def _sample_action_seq(self):
		if self.discrete:
			return np.random.choice(self.act_dim, (self.horizon, 
								 	   self.num_simulated_paths))
		else:
			return np.random.uniform(self.lb, 
									 self.ub, 
									 size=(self.horizon, 
									 	   self.num_simulated_paths, 
									 	   self.act_dim)
									 )

	def _sim_actions_forward(self, state):
		# First, we need to broadcast the state
		# from shape [obs_dim] to shape [num_simulated_paths, obs_dim].
		states = [np.broadcast_to(state,(self.num_simulated_paths, self.obs_dim))]

		# Next, get the open-loop action sequences.
		actions = self._sample_action_seq()

		# Now, we will use the dynamics model to predict the state sequence
		# which follows from executing those actions.
		curr_states = states[0]
		for i in range(self.horizon):
			curr_states = self.dyn_model.predict(curr_states, actions[i])
			states.append(curr_states)
		return states, actions

	def get_action(self, state):
		#start_time = time.time()
		states, actions = self._sim_actions_forward(state)
		# import IPython
		# IPython.embed()
		e = self.env
		while not hasattr(e, "rstate"):
			e = e.env
		rstate = np.concatenate([e.rstate] * actions.shape[1])
		costs = trajectory_cost_fn(self.cost_fn, states[:-1],
			actions[:-1], states[1:], rstate)
		best_simulated_path = np.argmax(costs)
		best_action = actions[:self.plan_every, best_simulated_path]
		#end_time = time.time()
		#print(end_time - start_time)
		return best_action