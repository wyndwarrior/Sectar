import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from traj2vec.utils.plot_utils import record_fig
from traj2vec.utils.torch_utils import from_numpy

class Dataset:
    def __init__(self, data_path, raw_data=None, data_size=-1, batch_size=32, normalize=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.normalize = normalize
        if raw_data is None:
            self.raw_data = np.load(data_path)
        else:
            self.raw_data = raw_data

        self.data_size = data_size
        if data_size != -1:
            self.raw_data = self.raw_data[:data_size]
        self.n = self.raw_data.shape[0]
        self.buffer = np.zeros(self.raw_data.shape)


        axis = tuple(range(len(self.raw_data.shape))[:-1])
        self.mean = self.raw_data.mean(axis=axis)
        self.std = self.raw_data.std(axis=axis)
        self.std[self.std == 0] = 1
        if normalize:
            self.raw_data = self.normalize_data(self.raw_data)

        self._train_data, self._train_target = None, None
        self._dataloader = None
        self._hard_idx = None



    def normalize_data(self, data):
        return (data - self.mean) / self.std

    def unnormalize(self, data):
        if self.normalize:
            if data.shape[-1] != self.mean.shape[-1]:
                orig_shape = data.shape
                reshape = data.reshape((orig_shape[0], -1, self.mean.shape[-1]))
                return ((reshape * self.std) + self.mean).reshape(orig_shape)
            else:
                return data * self.std + self.mean
        else:
            return data

    def setup_traindata(self):
        return self.raw_data

    def setup_train_target(self):
        # assume targets to be same as input by default for VAE
        return self.setup_traindata()

    def setup_dataloader(self):
        data = from_numpy(self.train_data.astype(np.float32))
        target = from_numpy(self.train_target.astype(np.float32))
        assert data.shape[0] >= self.batch_size, "Data size must be greater than batch size"
        return DataLoader(TensorDataset(data, target), batch_size=self.batch_size, shuffle=True,
                          drop_last=True)

    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = self.setup_traindata()
        return self._train_data

    @property
    def train_target(self):
        if self._train_target is None:
            self._train_target = self.setup_train_target()
        return self._train_target

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = self.setup_dataloader()
        return self._dataloader

    @property
    def input_dim(self):
        return self.train_data.shape[-1]

    def plot_compare(self, y, x, itr=0):
        pass

    def plot_pd_compare(self, traj_lst, traj_names, itr):
        pass

    def sample(self, batch_size):
        idx = np.random.randint(low=0, high=self.n, size=batch_size)
        return self.train_data[idx, ...], self.train_target[idx, ...]

    def sample_hard(self, batch_size, easy=False):
        if not hasattr(self, 'compute_hard_idx'):
            return self.sample(batch_size)
        if self._hard_idx is None:
            self._hard_idx = self.compute_hard_idx()
        if easy:
            idx = self._hard_idx[-batch_size:][::-1]
        else:
            idx = self._hard_idx[:batch_size]
        return self.train_data[idx, ...], self.train_target[idx, ...]

class TrajDataset(Dataset):
    def __init__(self, obs_dim, action_dim, path_len, env_id,
                 use_actions=False, flat_traj=True, **kwargs):
        """
        :param obs_dim:
        :param action_dim:
        :param path_len:
        :param env_id:
        :param use_actions:
        :param flat_traj: Where to flatten trajs into (bs, path_len * obs_dim)
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.path_len = path_len
        self.env_id = env_id
        self.use_actions = use_actions
        self.flat_traj = flat_traj

        # data should include 1 + path_len obs
        print(self.raw_data.shape)
        assert self.raw_data.shape[1] == path_len + 1
        #import pdb; pdb.set_trace()
        self.trajs = self.raw_data[:, :path_len+1, :]
        # self.flat_trajs = self.trajs.reshape((self.n, -1))

        self.obs = self.trajs[:, :, :self.obs_dim]
        # self.flat_obs = self.obs.reshape((self.n, -1))

        if env_id == 'TwoDMaze':
            self.axes = ([-.25, .25], [-.25, .25])
        else:
            self.axes = None

    def setup_traindata(self):
        output = self.trajs
        if not self.use_actions:
            output = self.obs

        if self.flat_traj:
            output = output.reshape((self.n, -1))

        return output

    def process(self, traj):
        # Process for plotting
        if len(traj.shape) == 2:
            return traj.reshape((traj.shape[0], self.path_len+1, -1))
        else:
            return traj

    def plot_all_traj(self, plot_traj, ax, color='purple'):
        traj = plot_traj
        if len(plot_traj.shape) < 2:
            traj = plot_traj.reshape((self.path_len+1, -1))
        ax.plot(traj[:, 0], traj[:, 1], color=color)
        if self.axes is not None:
            ax.set_xlim(self.axes[0])
            ax.set_ylim(self.axes[1])


    def plot_pd_compare(self, traj_lst, traj_names, itr, name='Full_State', save_dir='full_state'):

        f, axarr = plt.subplots(1, len(traj_lst), sharex=True, sharey=True)
        for ax, traj, traj_name in zip(axarr, traj_lst, traj_names):
            self.plot_all_traj(traj, ax)
            ax.set_title(traj_name)
        record_fig(name, save_dir, itr)

    def plot_hard_samples(self, hard_sd, easy_sd, hard_pd, easy_pd, itr, name='hard_samples', save_dir='hardveasy'):
        f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
        for traj in hard_sd:
            self.plot_all_traj(traj, axarr[0], color='red')
        for traj in easy_sd:
            self.plot_all_traj(traj, axarr[0], color='blue')
        for traj in hard_pd:
            self.plot_all_traj(traj, axarr[1], color='red')
        for traj in easy_pd:
            self.plot_all_traj(traj, axarr[1], color='blue')
        axarr[0].set_title('sd')
        axarr[1].set_title('pd')
        record_fig(name, save_dir, itr)

class BufferTrajDataset(TrajDataset):
    def __init__(self, buffer_size=10000, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        buffer_shape = tuple([buffer_size] + list(self.train_data.shape[1:]))
        self.train_buffer = np.zeros(buffer_shape)
        assert self.n <= buffer_size, "Buffer size should be bigger than data"
        self.train_buffer[:self.n] = self.train_data
        self.start = self.n

    def clear(self):
        self.start = 0
        self.n = 0

    def add_samples(self, samples):
        n_samples = samples.shape[0]
        start = self.start
        end = min(self.buffer_size, start + n_samples)
        n_add_end = end - start # Number of samples adding to end
        self.train_buffer[start:end, ...] = samples[:n_add_end, ...]

        # Handle wrap around
        if n_add_end < n_samples:
            self.train_buffer[:(n_samples - n_add_end)] = samples[n_add_end:, ...]
        self.n = min(self.buffer_size, self.n + n_samples)
        self.start = (start + n_samples) % self.buffer_size

        self._train_data = self.train_buffer[:self.n, ...]
        self._train_target = self._train_data
        self._dataloader = self.setup_dataloader()

class ButtonDataset(BufferTrajDataset):
    def plot_pd_compare(self, traj_lst, traj_names, itr, name='Full_State', save_dir='full_state'):
        f, axarr = plt.subplots(1, len(traj_lst))
        for ax, traj, traj_name in zip(axarr, traj_lst, traj_names):
            #import pdb; pdb.set_trace()
            ax.imshow(traj.reshape((-1, self.path_len)))
            ax.set_title(traj_name)
            #break
        record_fig(name, save_dir, itr)

    def process(self, traj):

        actions = traj.reshape((traj.shape[0], self.path_len, -1)).argmax(-1)
        idx = np.expand_dims(np.repeat(np.expand_dims(np.arange(0, self.path_len), 0), traj.shape[0], 0), -1)
        return np.concatenate([idx, np.expand_dims(actions, -1)], -1)

class WheeledContDataset(BufferTrajDataset):
    def __init__(self, pltidx=[0, 1], **kwargs):
        super().__init__(**kwargs)
        self.pltidx = pltidx

    def process(self, traj):
        # Process for plotting
        if len(traj.shape) == 2:
            return traj.reshape((traj.shape[0], self.path_len, -1))[:, :, :2]
        else:
            return traj[:, :, :2]

    def plot_all_traj(self, plot_traj, ax, color=None):
        traj = plot_traj
        if len(plot_traj.shape) < 2:
            traj = plot_traj.reshape((self.path_len, -1))

        # colors = ['purple', 'magenta', 'green', 'black', 'yellow']
        # if color is not None:
        #     colors[0] = color
        # num_moving = self.num_obj() + 1
        # for i in range(num_moving):
        if len(traj.shape) == 2:
            ax.plot(traj[:, self.pltidx[0]], traj[:, self.pltidx[1]])
        else:
            for t in traj:
                ax.plot(t[:, self.pltidx[0]], t[:, self.pltidx[1]])
        #     ax.scatter(traj[-1, i*2], traj[-1, i*2+1], color=colors[i], s = 10)

        # picked_idx = traj[:, num_moving*2:].argmax(axis=-1)
        # for i in range(num_moving):
        #     obj_pos = picked_idx == i
        #     ax.scatter(traj[:, 0][obj_pos], traj[:, 1][obj_pos], facecolors='none', edgecolors=colors[i])


    def plot_pd_compare(self, traj_lst, traj_names, itr, name='Full_State', save_dir='full_state', goals=None, goalidx=None):
        # colors = ['magenta', 'green', 'black', 'yellow']
        if goals is not None:
            import matplotlib.cm as cm
            colors = cm.rainbow(np.linspace(0, 1, goals.shape[0]))
            goals = goals[goalidx:]
            colors = colors[goalidx:]
        mult = 5
        f, axarr = plt.subplots(1, len(traj_lst), sharex=True, sharey=True, figsize=(len(traj_lst) * mult, mult))
        for ax, traj, traj_name in zip(axarr, traj_lst, traj_names):
            self.plot_all_traj(traj, ax)
            if goals is not None:
                for pt, color in zip(goals.reshape((-1, 2)), colors):
                    ax.scatter(pt[0], pt[1], color=color, marker='x')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title(traj_name)
        record_fig(name, save_dir, itr)


    def num_obj(self):
        return 4

class PlayPenContDataset(BufferTrajDataset):

    def process(self, traj):
        # Process for plotting
        if len(traj.shape) == 2:
            return traj.reshape((traj.shape[0], self.path_len, -1))[:, :, :2]
        else:
            return traj[:, :, :2]

    def plot_all_traj(self, plot_traj, ax, color=None):
        traj = plot_traj
        if len(plot_traj.shape) < 2:
            traj = plot_traj.reshape((self.path_len, -1))

        colors = ['purple', 'magenta', 'green', 'black', 'yellow']
        if color is not None:
            colors[0] = color
        num_moving = self.num_obj() + 1
        if len(plot_traj.shape) == 2: 
            plot_traj = [plot_traj]
        for traj in plot_traj:
            for i in range(num_moving):
                if i * 2 + 1 < traj.shape[1]:
                    ax.plot(traj[:, i*2], traj[:, i*2+1], color=colors[i])
                    ax.scatter(traj[-1, i*2], traj[-1, i*2+1], color=colors[i], s = 100)

            if num_moving*2 < traj.shape[1]:
                picked_idx = traj[:, num_moving*2:].argmax(axis=-1)
                for i in range(num_moving):
                    obj_pos = picked_idx == i
                    ax.scatter(traj[:, 0][obj_pos], traj[:, 1][obj_pos], facecolors='none', edgecolors=colors[i])


    def plot_pd_compare(self, traj_lst, traj_names, itr, name='Full_State', save_dir='full_state', goals=None, goalidx=None):
        colors = ['magenta', 'green', 'black', 'yellow']
        if goals is not None:
            if goals.reshape((-1, 2)).shape[0] > len(colors):
                import matplotlib.cm as cm
                colors = cm.rainbow(np.linspace(0, 1, goals.shape[0]))
                goals = goals[goalidx:]
                colors = colors[goalidx:]
        f, axarr = plt.subplots(1, len(traj_lst), sharex=True, sharey=True)
        for ax, traj, traj_name in zip(axarr, traj_lst, traj_names):
            self.plot_all_traj(traj, ax)
            if goals is not None:
                for pt, color in zip(goals.reshape((-1, 2)), colors):
                    ax.scatter(pt[0], pt[1], color=color, marker='x')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title(traj_name)
        record_fig(name, save_dir, itr)

    def num_obj(self):
        return 4

class PlayPenSingleDataset(PlayPenContDataset):
    def num_obj(self):
        return 1

    def compute_hard_idx(self):
        #import pdb; pdb.set_trace()
        obj_dist = np.square(self.train_data[:, -self.obs_dim:][:, 2:4] - np.array([1.6, 0])).mean(-1)
        return obj_dist.argsort()[::-1]


class DACOPlaypenDataset(BufferTrajDataset):

    def process(self, traj):
        # Process for plotting
        if len(traj.shape) == 2:
            return traj.reshape((traj.shape[0], self.path_len, -1))[:, :, :2]
        else:
            return traj[:, :, :2]

    def plot_all_traj(self, plot_traj, ax, color=None):
        traj = plot_traj
        if len(plot_traj.shape) < 2:
            traj = plot_traj.reshape((self.path_len, -1))

        colors = ['purple', 'magenta', 'green', 'black', 'yellow']
        if color is not None:
            colors[0] = color
        for i in range(5):
            if i * 2 + 1 < traj.shape[1]:
                ax.plot(traj[:, i*2], traj[:, i*2+1], color=colors[i])

        # Plot picked up obj


    def plot_pd_compare(self, traj_lst, traj_names, itr, name='Full_State', save_dir='full_state'):

        f, axarr = plt.subplots(1, len(traj_lst), sharex=True, sharey=True)
        for ax, traj, traj_name in zip(axarr, traj_lst, traj_names):
            self.plot_all_traj(traj, ax)
            ax.set_title(traj_name)
        record_fig(name, save_dir, itr)

class ReacherDataset(TrajDataset):
    def __init__(self, obs_dim, action_dim, path_len, env_id,
                 use_actions=False, flat_traj=True, **kwargs):
        """
        :param obs_dim:
        :param action_dim:
        :param path_len:
        :param env_id:
        :param use_actions:
        :param flat_traj: Where to flatten trajs into (bs, path_len * obs_dim)
        :param kwargs:
        """
        super(TrajDataset, self).__init__(**kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.path_len = path_len
        self.env_id = env_id
        self.use_actions = use_actions
        self.flat_traj = flat_traj

        self.trajs = self.raw_data

        # Add back goal vec to get fingertip position and set goal to 0
        self.trajs[:, :, 8:10] += self.trajs[:, :, 4:6]
        self.trajs[:, :, 4:6] = 0


        self.flat_trajs = self.trajs.reshape((self.n, -1))

        self.obs = self.trajs[:, :, :self.obs_dim]
        self.flat_obs = self.obs.reshape((self.n, -1))

    def process_traj(self, traj):
        # Process for plotting
        if len(traj.shape) == 2:
            plot_traj = traj.reshape((traj.shape[0], self.path_len, -1))
        else:
            plot_traj = traj
        return plot_traj[:, :, 8:10]


class DiscreteTrajDataset(TrajDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DiscretePlayPenDataset(DiscreteTrajDataset):
    def __init__(self, only_use_agent_pos=True, **kwargs):
        super().__init__(**kwargs)
        self.x_bounds = [-2.8, 3.2]
        self.y_bounds = [-2.8, 3.2]
        self.l_bounds = np.array([self.x_bounds[0], self.y_bounds[0]])
        self.step_size = 0.2

        self.x_range = self.x_bounds[1] - self.x_bounds[0]
        self.y_range = self.y_bounds[1] - self.y_bounds[0]

        self.x_size = int(self.x_range // self.step_size + 1)
        self.y_size = int(self.y_range // self.step_size + 1)

        self.x_lin = np.arange(self.x_size) * self.step_size - self.x_bounds[1]
        self.y_lin = np.arange(self.y_size) * self.step_size - self.y_bounds[1]

        self.grid_size = int(self.x_size * self.y_size)

        # One hot size for each env dim
        # [agent_pos, ob1 pos, ... obj 4 pos, attached obj]
        # Pos can vary up to x and y size for agent and 4 objects.
        # Attached obj is 0 for nothing and 1-4 for objects.
        self.all_cat_sizes = [self.x_size, self.y_size] * 5 + [5]
        if only_use_agent_pos:
            self._cat_sizes = [self.x_size, self.y_size] # Only use agent position for now
        else:
            self._cat_sizes = self.all_cat_sizes

        self.total_cat_sizes = sum(self.all_cat_sizes)
        self.obs = self.obs[:, :self.path_len, :]

    def create_one_hot(self, traj):
        # traj is (batch_size, path_len, obs_dim)
        n = traj.shape[0]

        pos = traj[:, :, :-2]
        pos_idx = self.pos_to_idx(pos.reshape((-1, 2))).flatten().astype(np.int32)
        one_hot_pos = np.zeros((pos_idx.size, self.all_cat_sizes[0]))
        one_hot_pos[np.arange(pos_idx.size), pos_idx] = 1
        one_hot_pos = one_hot_pos.reshape((n, self.path_len, sum(self.all_cat_sizes[:-1])))


        obj = traj[:, :, -2:]
        obj_idx = (obj[:, :, 0] + 1).flatten().astype(np.int32)
        one_hot_obj = np.zeros((obj_idx.size, self.all_cat_sizes[-1]))
        one_hot_obj[np.arange(obj_idx.size), obj_idx] = 1
        one_hot_obj = one_hot_obj.reshape((n, self.path_len, self.all_cat_sizes[-1]))

        one_hot = np.concatenate([one_hot_pos, one_hot_obj], -1)
        all_idx = np.concatenate([pos_idx.reshape((n, self.path_len, -1)),
                                  obj_idx.reshape((n, self.path_len, -1))], -1)

        # Only use agent position for now
        one_hot = one_hot[:, :, :sum(self.cat_sizes)]
        return one_hot, all_idx


    def setup_traindata(self):
        # train data will be (n, path_len, sum(cat_sizes))
        # targets will be (n, path_len, len(cat_sizes))
        one_hot, all_idx = self.create_one_hot(self.obs)
        self._train_target = all_idx

        return one_hot.reshape((one_hot.shape[0], -1))


    @property
    def cat_sizes(self):
        return self._cat_sizes

    def pos_to_idx(self, coord):
        # Coord is 2d np array (bs, 2)
        idx = (coord - self.l_bounds) // self.step_size
        return idx

    def idx_to_pos(self, idx):
        pos = (idx * self.step_size) + self.l_bounds
        return pos
