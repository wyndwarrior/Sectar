import numpy as np
import torch
import traj2vec.utils.logger as logger
from torch.optim import Adam
from traj2vec.math.distributions.normal import Normal
from traj2vec.utils.plot_utils import plot_traj_sets
from traj2vec.utils.torch_utils import get_numpy, FloatTensor
from traj2vec.utils.torch_utils import gpu_enabled, Variable, np_to_var, from_numpy


class VAEBC:
    def __init__(self, encoder, decoder, latent_dim, step_dim, obs_dim, act_dim, policy, env, optimizer=None, loss_type='mse',
                 init_kl_weight=.001, max_kl_weight=.1, kl_mul=1.07, vae_loss_weight=1, lr=1e-3, bc_weight=100, ego=False, egoidx=None):
        self.encoder = encoder
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.decoder = decoder
        self.ego = ego
        self.egoidx = egoidx
        self.bc_weight = bc_weight
        self.env = env()
        self.policy = policy
        self.unit_n = Normal(Variable(torch.zeros(1, latent_dim)),
                             log_var=Variable(torch.zeros(1, latent_dim)))
        self.latent_dim = latent_dim
        self.step_dim = step_dim
        self.init_kl_weight = init_kl_weight
        self.max_kl_weight = max_kl_weight
        self.kl_mul = kl_mul
        if optimizer is None:
            optimizer = Adam(self.get_params(), lr=lr, eps=1e-5)
        self.loss_type = loss_type 
        self.vae_loss_weight = vae_loss_weight
        self.optimizer = optimizer
        if gpu_enabled():
            self.encoder.cuda()
            self.decoder.cuda()

    def get_params(self):
        return self.encoder.get_params() + self.decoder.get_params() + self.policy.get_params()

    def encode(self, x):
        if self.encoder.recurrent():
            x = x.view((x.size()[0], -1, self.step_dim)).clone()
            if self.ego:
                x[:, :, self.egoidx] -= x[:, :1, self.egoidx]
            x = x.transpose(0, 1)
        return self.encoder.forward(x)

    def decode(self, x, z):
        if self.decoder.recurrent():
            initial_input = x[:, :self.step_dim].contiguous().clone()
            if self.ego:
                diff = initial_input[:, self.egoidx].clone()
                initial_input[:, self.egoidx] = 0
            output = self.decoder.forward(z, initial_input=Variable(initial_input))
            if self.ego:
                bs = output.mean.shape[0]
                mean = output.mean.view((bs, -1, self.step_dim))
                mean[:, :, self.egoidx] += Variable(diff[:, None])
                output.mean = mean.view(output.mean.shape)
            return output
        else:
            return self.decoder.forward(z)

    def save(self, snapshot_dir, itr):
        import os
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        torch.save(self.encoder.state_dict_lst(), snapshot_dir + '/encoder_%d.pkl' %itr)
        torch.save(self.policy.state_dict_lst(), snapshot_dir + '/policy_%d.pkl' %itr)
        torch.save(self.decoder.state_dict_lst(), snapshot_dir + '/decoder_%d.pkl' % itr)

    def load(self, snapshot_dir, itr):
        self.policy.load_state_dict(torch.load(snapshot_dir + '/policy_%d.pkl' %itr))
        self.encoder.load_state_dict(torch.load(snapshot_dir + '/encoder_%d.pkl' %itr))
        self.decoder.load_state_dict(torch.load(snapshot_dir + '/decoder_%d.pkl' % itr))

    def loss(self, x, y_dist, z_dist):
        mse = torch.pow(y_dist.mle - x, 2).mean(-1)
        neg_ll = - y_dist.log_likelihood(x) / self.decoder.path_len
        kl = z_dist.kl(self.unit_n)
        return mse, neg_ll, kl

    def compute_kl_weight(self, itr):
        return float(min(self.max_kl_weight, self.init_kl_weight * np.power(self.kl_mul, itr)))

    def compute_loss(self, x, actdata, z, z_dist):
        y_dist = self.decode(x, z)
        y = x[:, self.step_dim:].contiguous()

        xview = x.view((x.size()[0], -1, self.obs_dim)).clone()
        if self.ego:
            xview[:, :, self.egoidx] -= xview[:, [0], self.egoidx].unsqueeze(1)
        zexpand = z.unsqueeze(1).expand(*xview.size()[:2], z.size()[-1])
        xz = torch.cat(( Variable(xview), zexpand), -1)
        xz_view = xz.view((-1, xz.size()[-1]))
        if self.policy.recurrent():
            self.policy.reset(x.size()[0])
            xz_view = xz.transpose(0, 1)
            actdata = actdata.view((actdata.size()[0], -1, self.act_dim))
            actdata = actdata.transpose(0, 1).contiguous()
        dist = self.policy.forward(xz_view)
        act_view = actdata.view((-1, self.act_dim))
        
        bcloss = -dist.log_likelihood(Variable(act_view))
        if self.policy.recurrent():
            bcloss = bcloss.view(*actdata.size()[:2]).mean(0)
        else:
            bcloss = bcloss.view((actdata.size()[0], -1)).mean(1)

        mse, neg_ll, kl = self.loss(Variable(y), y_dist, z_dist)
        return mse, neg_ll, kl, bcloss, z_dist

    def forward_batch(self, batch):
        obsdata, actdata = self.splitobs(batch)

        x = obsdata
        z_dist = self.encode(Variable(x))
        z = z_dist.sample()
        return self.compute_loss(x, actdata, z, z_dist)

    def loss_generator(self, dataset):
        kl_weight = self.compute_kl_weight(0)
        for batch_idx, (batch, target) in enumerate(dataset.dataloader):
            mse, neg_ll, kl, bcloss, z_dist = self.forward_batch(batch)
            mse = mse.mean(0)
            neg_ll = neg_ll.mean(0)
            kl = kl.mean(0)
            bcloss = bcloss.mean(0)
            if self.loss_type == 'mse':
                loss = self.vae_loss_weight * mse + kl_weight * kl + bcloss * self.bc_weight
            else:
                loss = self.vae_loss_weight * neg_ll + kl_weight * kl + bcloss * self.bc_weight

            stats = {
                'MSE': mse,
                'Total Loss': loss,
                'LL': neg_ll,
                'KL Loss': kl,
                'BC Loss': bcloss
            }
            yield loss, stats

    def train_epoch(self, dataset, epoch=0, train=True, max_steps=1e99):
        full_stats = dict([('MSE',0), ('Total Loss', 0), ('LL', 0), ('KL Loss', 0),
                           ('BC Loss', 0)])

        n_batch = 0
        self.optimizer.zero_grad()
        for loss, stats in self.loss_generator(dataset):
            if train:
                loss.backward()
                self.optimizer.step()

            for k in stats.keys():
                full_stats[k] += get_numpy(stats[k])[0]
            n_batch += 1
            if n_batch >= max_steps:
                break
            self.optimizer.zero_grad()

        for k in full_stats.keys():
            full_stats[k] /= n_batch

        return full_stats

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

    def print_diagnostics(self, stats):
        for k in sorted(stats.keys()):
            logger.log('%s: %f' %(k, stats[k]))

    def log_diagnostics(self, stats):
        for k in sorted(stats.keys()):
            logger.record_tabular(k, stats[k])

    def plot_compare(self, dataset, itr, save_dir='trajs'):


        x = FloatTensor(dataset.sample_hard(5)[0])
        x, actdata = self.splitobs(x)
        target = x[:, self.step_dim:]

        y_dist = self.decode(x, self.encode(Variable(x)).sample())

        traj_sets = [dataset.unnormalize(get_numpy(traj_set)) for traj_set in [target, y_dist.mle]]

        traj_names = ['expert', 'sd']
        plot_traj_sets([dataset.process(traj_set) for traj_set in traj_sets], traj_names, itr, env_id=dataset.env_id)
        for traj_no in range(5):
            dataset.plot_pd_compare([x[traj_no, ...] for x in traj_sets], traj_names, itr, name='Full_State_%d' % traj_no,
                                    save_dir=save_dir)


    def plot_interp(self, dataset, itr):
        x = dataset.sample(2)[0]

        x1 = np_to_var(np.expand_dims(x[0, ...], 0))
        x2 = np_to_var(np.expand_dims(x[1, ...], 0))

        l1 = get_numpy(self.encode(x1).sample(deterministic=True))
        l2 = get_numpy(self.encode(x2).sample(deterministic=True))

        num_interp = 7
        latents = np.zeros((num_interp, self.latent_dim))
        for i in range(self.latent_dim):
            latents[:, i] = np.interp(np.linspace(0, 1, num_interp), [0, 1], [l1[0, i], l2[0, i]])


        traj = dataset.unnormalize(get_numpy(self.decode(x1.repeat(num_interp, 1).data, np_to_var(latents)).mle))

        traj_sets = [traj[i, ...] for i in range(num_interp)]
        traj_names = range(num_interp)
        dataset.plot_pd_compare(traj_sets, traj_names, itr, save_dir='interp')


    def plot_random(self, dataset, itr, sample_size=5):
        y_dist, latent = self.sample(dataset, sample_size)

        traj_sets = [dataset.unnormalize(get_numpy(y_dist.mle))]
        traj_names = ['sampled']
        plot_traj_sets([dataset.process(traj_set) for traj_set in traj_sets], traj_names, itr, figname='sampled',
                       env_id=dataset.env_id)

    def splitobs(self, x):
        N = x.size()[0]
        x = x.view((N, -1, self.obs_dim + self.act_dim))
        obsdata = x[:, :, :self.obs_dim].contiguous()
        actdata = x[:, :, self.obs_dim:].contiguous()
        return obsdata.view((N, -1)), actdata.view((N, -1))

    def test(self, dataset):
        data = FloatTensor(dataset.train_data)
        x, actdata = self.splitobs(data)
        y = x[:, self.step_dim:]
        z_dist = self.encode(Variable(x))
        z = z_dist.sample()
        y_dist = self.decode(x, z)
        log_likelihood = torch.pow(y_dist.mle - Variable(y), 2).mean(-1).mean(0)
        return get_numpy(log_likelihood)[0]


    def sample(self, dataset, sample_size):
        trajs, _ = dataset.sample(sample_size)
        latent = Variable(torch.randn((sample_size, self.latent_dim)))
        return self.decode(FloatTensor(trajs), latent), latent

    def test_pd(self, dataset, lim=-1):
        def rollout(env, policy, max_path_length, add_input=None, volatile=False, reset_args = None):
            sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[])
            obs = env.reset(reset_args)
            for s in range(max_path_length):
                policy_input = Variable(from_numpy(np.array([obs])).float(), volatile=volatile)

                if add_input is not None:
                    policy_input = torch.cat([policy_input, add_input], -1)
                if s == 0:
                    policy.reset(1)
                if policy.recurrent():
                    policy_input = policy_input.unsqueeze(0)
                action_dist = policy.forward(policy_input)
                action = action_dist.sample()

                x = env.step(get_numpy(action))
                next_obs = x[0]
                sd['obs'].append(obs)
                sd['rewards'].append(x[1])
                sd['actions'].append(action)
                obs = next_obs
            sd['obs'].append(obs)
            sd['obs'] = np.array(sd['obs']) # (bs, max_path_length, obs_dim)
            sd['rewards'] = np.array(sd['rewards']) # (bs, max_path_length)
            sd['actions'] = torch.stack(sd['actions'], 1)

            return sd
        errs = []
        limdata = dataset.train_data
        if lim != -1:
            limdata = dataset.train_data[:lim]
        for data in limdata:
            traj = data.reshape((-1, self.act_dim + self.obs_dim))[None, :, :self.obs_dim]
            N = traj.shape[1]
            x = np_to_var(traj)

            z_dist = self.encode(x)
            z = z_dist.sample()
            ro = rollout(self.env, self.policy, N, z, reset_args=traj[0, 0])
            errs.append(np.linalg.norm(traj- ro['obs'][:N]))

        return np.mean(errs)

    def train(self, dataset, test_dataset=None, max_epochs=10000, save_step=1000, print_step=1,  plot_step=1,
              record_stats=False):

        for epoch in range(1, max_epochs + 1):
            stats = self.train_epoch(dataset, epoch)
            test = self.train_epoch(test_dataset, epoch, train=False)
            for k, v in test.items():
                stats['V ' + k] = v
            stats['Test RL'] = self.test_pd(test_dataset)

            if epoch % print_step == 0:
                with logger.prefix('itr #%d | ' % epoch):
                    self.print_diagnostics(stats)

            if epoch % plot_step == 0:
                self.plot_compare(dataset, epoch)
                self.plot_interp(dataset, epoch)
                self.plot_compare(test_dataset, epoch, save_dir='test')
                self.plot_random(dataset, epoch)

            if epoch % save_step == 0 and logger.get_snapshot_dir() is not None:
                self.save(logger.get_snapshot_dir() + '/snapshots/', epoch)

            if record_stats:
                with logger.prefix('itr #%d | ' % epoch):
                    self.log_diagnostics(stats)
                    logger.dump_tabular()

        return stats


class TrajVAEBC(VAEBC):
    def __init__(self, path_len, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.path_len = path_len
        self.feature_dim = feature_dim
