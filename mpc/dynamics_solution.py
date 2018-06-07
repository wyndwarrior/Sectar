import tensorflow as tf
import numpy as np

def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=64, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

def normalize(x, x_mean, x_std):
    return (x - x_mean)/(x_std + 1e-8)

def unnormalize(x, x_mean, x_std):
    return x*x_std + x_mean

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 # neural network architecture options
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 # normalization statistics
                 normalization,
                 # training arguments
                 batch_size,
                 iterations,
                 learning_rate
                 ):
        self.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "shape"):
            self.act_dim = env.action_space.shape[0]
        else:
            self.act_dim = env.action_space.n

        # Data inputs
        self.obs_ph = tf.placeholder(shape=[None, self.obs_dim], name="obs", dtype=tf.float32)
        self.deltas = tf.placeholder(shape=[None, self.obs_dim], name="delta", dtype=tf.float32)
        self.act_ph = tf.placeholder(shape=[None, self.act_dim], name="action", dtype=tf.float32)

        # Normalize versions of the data inputs
        # (Where means / stds are computed outside of the NNDynamicsModel)
        normed_obs = normalize(self.obs_ph, normalization[0], normalization[1])
        normed_deltas = normalize(self.deltas, normalization[2], normalization[3])
        normed_act = normalize(self.act_ph, normalization[4], normalization[5])

        # The dynamics model takes (state, action) pairs as inputs,
        # so we concatenate the state and action vectors
        net_input = tf.concat([normed_obs, normed_act], axis=1)

        self.net = build_mlp(input_placeholder=net_input, 
                             output_size=self.obs_dim, 
                             scope="dyn_model", 
                             n_layers=n_layers, 
                             size=size, 
                             activation=activation, 
                             output_activation=output_activation
                             )
        # We'll presume that the network output is normalized, and now 
        # we'll de-normalize it so that it has the correct magnitude
        self.unnormalized_net = unnormalize(self.net, normalization[2], normalization[3])

        self.loss = tf.reduce_mean(tf.square(self.net - normed_deltas))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Save training arguments
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = None

    def fit(self, data, test_data):
        obs, acts, obs2 = data['observations'], data['actions'], data['next_observations']
        n_datapoints = len(obs)

        losses = []
        idx = np.asarray(range(len(obs)))
        num_batches = len(obs) // self.batch_size
        for itr in range(self.iterations):
            iteration_loss = 0
            np.random.shuffle(idx)
            for bn in range(num_batches):
                idxs = idx[bn*self.batch_size: (bn+1)*self.batch_size]
                feed_dict = {
                    self.obs_ph : obs[idxs],
                    self.act_ph : acts[idxs],
                    self.deltas : obs2[idxs]-obs[idxs],
                    }
                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict)
                if np.isnan(loss):
                    import IPython
                    IPython.embed()
                iteration_loss += loss
            losses.append(iteration_loss)
            test_loss = self.prediction_error(test_data)
            print("itr %d \t dynamics loss %.3f \t test loss %.3f"%(itr, iteration_loss, test_loss))
        return losses

    def predict(self, states, actions):
        if len(actions.shape) == 1:
            oh_acts = np.zeros((actions.shape[0], self.act_dim))
            oh_acts[range(actions.shape[0]), actions] = 1
            actions = oh_acts
        feed_dict = {self.obs_ph: states, self.act_ph: actions}
        return states + self.sess.run(self.unnormalized_net, feed_dict)

    def prediction_error(self, data):
        return self.sess.run([self.loss], {
                    self.obs_ph : data['observations'],
                    self.act_ph : data['actions'],
                    self.deltas : data['next_observations'] - data['observations']
                    })[0]

    def n_step_error(self, paths, n=10):
        all_obs_seq = np.zeros([n+1, len(paths), self.obs_dim])
        all_act_seq = np.zeros([n, len(paths), self.act_dim])
        for j, path in zip(range(len(paths)),paths):
            t = np.random.randint(0,len(path['observations'])-n)
            obs_seq = path['observations'][t:t+n+1]
            act_seq = path['actions'][t:t+n]
            all_obs_seq[:, j] = obs_seq
            all_act_seq[:, j] = act_seq

        all_next_obs_seq = np.zeros([n, len(paths), self.obs_dim])
        for k in range(n):
            next_obs = self.predict(all_obs_seq[k], all_act_seq[k])
            all_next_obs_seq[k,:] = next_obs

        l2_error = np.mean(np.square(all_next_obs_seq - all_obs_seq[1:]))
        return l2_error