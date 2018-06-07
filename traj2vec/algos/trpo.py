import numpy as np
import torch
import torch.autograd as autograd
import traj2vec.utils.logger as logger
from traj2vec.algos.batch_polopt import BatchPolopt
from traj2vec.math.misc import krylov
from traj2vec.utils.torch_utils import from_numpy, Variable, np_to_var, get_numpy


class PerlmutterHvp(object):
    def __init__(self, num_slices=1):
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params()

        constraint_grads = autograd.backward(f, params)
        for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
            if grad is None:
                constraint_grads[idx] = Variable(torch.zeros(param.size()), requires_grad=True)

        def Hx_plain(xs):
            Hx_plain_splits = autograd.backward(
                torch.sum(
                    torch.stack([torch.sum(g * x) for g, x in zip(constraint_grads, xs)])
                ),
                params
            )
            for idx, (Hx, param) in enumerate(zip(Hx_plain_splits, params)):
                if Hx is None:
                    Hx_plain_splits[idx] = torch.zeros_like(param)
            return [x.view(-1) for x in Hx_plain_splits]

        self.f_Hx_plain = Hx_plain

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = self.f_Hx_plain
            return ret

        return eval

class TRPO(BatchPolopt):
    """Natural Policy Optimization"""
    def __init__(
            self,
            env,
            policy,
            baseline,
            step_size=0.01,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            accept_violation=False,
            hvp_approach=None,
            num_slices=1,
            **kwargs):

        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices
        self._constraint_name = 'mean_kl'
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach
        self._max_constraint_val = step_size
        self._target = self.policy


        super().__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    def compute_loss_terms(self, sd):

        obs = np_to_var(sd['obs_flat'])
        old_dist = sd['action_dist'].detach().reshape((-1, sd['action_dist'].dim))

        new_dist = self.policy.forward(obs)

        mean_kl = old_dist.kl(new_dist).mean(0)
        lr = old_dist.log_likelihood_ratio(sd['actions'].view(-1, old_dist.dim), new_dist)

        surr_loss = -(lr.view(sd['discount_adv'].shape) * np_to_var(sd['discount_adv']))
        surr_loss = surr_loss.sum(-1).mean(0)

        return surr_loss, mean_kl

    def loss(self, sd):
        loss = -(sd['log_prob'] * np_to_var(sd['discount_adv']))
        return loss.sum(-1).mean(0)

    def optimize_policy(self, itr, samples_data):
        prev_param = get_numpy(self._target.get_params_flat())

        self.policy.zero_grad()
        loss_before = self.loss(samples_data)
        loss_before.backward()
        flat_g = self.policy.get_params_flat()
        loss_before = get_numpy(loss_before)[0]

        Hx = self._hvp_approach.build_eval(samples_data)

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_params_flat(from_numpy(cur_param))
            loss, constraint_val = self.compute_loss_terms(samples_data)

            if self._debug_nan and np.isnan(constraint_val):
                import ipdb;
                ipdb.set_trace()
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
            self._max_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" % self._constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                logger.log("Violated because constraint %s is violated" % self._constraint_name)
            self._target.set_param_values(prev_param, trainable=True)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")