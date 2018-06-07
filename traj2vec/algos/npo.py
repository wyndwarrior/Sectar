import numpy as np
import scipy.optimize
import torch
import traj2vec.utils.logger as logger
from traj2vec.algos.batch_polopt import BatchPolopt
from traj2vec.utils.torch_utils import from_numpy, np_to_var, get_numpy


class NPO(BatchPolopt):
    """Natural Policy Optimization"""
    def __init__(
            self,
            env,
            policy,
            baseline,
            step_size=0.01,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_penalty_itr=10,
            adapt_penalty=True,
            **kwargs):

        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._max_penalty_itr = max_penalty_itr
        self._adapt_penalty = adapt_penalty
        self._constraint_name = 'mean_kl'
        self._max_constraint_val = step_size

        super().__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    def get_opt_output(self, sd, penalty):
        self.policy.zero_grad()

        penalized_loss, surr_loss, mean_kl = self.compute_loss_terms(sd, penalty)
        penalized_loss.backward()
        params = self.policy.get_params()
        grads = [p.grad for p in params]

        flat_grad = torch.cat([g.view(-1) for g in grads])
        #import pdb; pdb.set_trace()
        return [get_numpy(penalized_loss.double())[0], get_numpy(flat_grad.double())]

    def compute_loss_terms(self, sd, penalty):

        obs = sd['obs_flat_var']
        old_dist = sd['action_dist_flat'] #sd['action_dist'].detach().reshape((-1, sd['action_dist'].dim))
        actions = sd['actions_flat']
        if 'add_input_flat' in sd:
            add_input = sd['add_input_flat']
            new_dist = self.policy.forward(torch.cat([obs, add_input], -1))
        else:
            new_dist = self.policy.forward(obs)


        mean_kl = old_dist.kl(new_dist).mean(0)

        lr = old_dist.log_likelihood_ratio(actions, new_dist)

        surr_loss = -(lr.view(sd['discount_adv'].shape) * sd['discount_adv_var'])
        surr_loss = surr_loss.sum(-1).mean(0)

        penalized_loss = surr_loss + penalty * mean_kl

        return penalized_loss, surr_loss, mean_kl


    def optimize_policy(self, itr, samples_data):
        try_penalty = float(np.clip(
            self._penalty, self._min_penalty, self._max_penalty))
        penalty_scale_factor = None

        def gen_f_opt(penalty):
            def f(flat_params):
                self.policy.set_params_flat(from_numpy(flat_params))
                return self.get_opt_output(samples_data, penalty)
            return f

        cur_params = get_numpy(self.policy.get_params_flat().double())
        opt_params = cur_params

        # Save views of objs for efficiency
        samples_data['obs_flat_var'] = np_to_var(samples_data['obs_flat'])
        samples_data['action_dist_flat'] = samples_data['action_dist'].detach().reshape((-1, samples_data['action_dist'].dim))
        samples_data['actions_flat'] = samples_data['actions'].view(-1, self.action_dim)
        samples_data['discount_adv_var'] = np_to_var(samples_data['discount_adv'])

        for penalty_itr in range(self._max_penalty_itr):
            logger.log('trying penalty=%.3f...' % try_penalty)

            itr_opt_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
                func=gen_f_opt(try_penalty), x0=cur_params,
                maxiter=self._max_opt_itr
            )

            _, try_loss, try_constraint_val = self.compute_loss_terms(samples_data, try_penalty)
            try_loss = get_numpy(try_loss)[0]
            try_constraint_val = get_numpy(try_constraint_val)[0]

            logger.log('penalty %f => loss %f, %s %f' %
                       (try_penalty, try_loss, self._constraint_name, try_constraint_val))

            if try_constraint_val < self._max_constraint_val or \
                    (penalty_itr == self._max_penalty_itr - 1 and opt_params is None):
                opt_params = itr_opt_params

            if not self._adapt_penalty:
                break

            # Decide scale factor on the first iteration, or if constraint violation yields numerical error
            if penalty_scale_factor is None or np.isnan(try_constraint_val):
                # Increase penalty if constraint violated, or if constraint term is NAN
                if try_constraint_val > self._max_constraint_val or np.isnan(try_constraint_val):
                    penalty_scale_factor = self._increase_penalty_factor
                else:
                    # Otherwise (i.e. constraint satisfied), shrink penalty
                    penalty_scale_factor = self._decrease_penalty_factor
                    opt_params = itr_opt_params
            else:
                if penalty_scale_factor > 1 and \
                                try_constraint_val <= self._max_constraint_val:
                    break
                elif penalty_scale_factor < 1 and \
                                try_constraint_val >= self._max_constraint_val:
                    break
            try_penalty *= penalty_scale_factor
            try_penalty = float(np.clip(try_penalty, self._min_penalty, self._max_penalty))
            self._penalty = try_penalty

        self.policy.set_params_flat(from_numpy(opt_params))