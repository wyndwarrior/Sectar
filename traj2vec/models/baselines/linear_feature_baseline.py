import numpy as np
import torch
from traj2vec.models.baselines.baseline import Baseline
from traj2vec.utils.torch_utils import np_to_var, Variable


class LinearFeatureBaseline(Baseline):
    def __init__(self, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff


    def _features(self, obs):
        o = np.clip(obs, -10, 10)
        l = o.shape[0]
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)


    def fit(self, obs_np, returns_np):
        bs, path_len, obs_dim = obs_np.shape
        obs = obs_np.reshape(-1, obs_dim)
        returns = returns_np.reshape(-1)

        featmat = self._features(obs)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, obs_np):
        bs, path_len, obs_dim = obs_np.shape
        obs = obs_np.reshape(-1, obs_dim)
        if self._coeffs is None:
            return Variable(torch.zeros((bs, path_len)))
        returns = self._features(obs).dot(self._coeffs).reshape((-1, path_len))
        return np_to_var(returns)