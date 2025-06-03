from flax import linen as nn
from flax.linen.initializers import constant
import distrax
from jax import numpy as jnp
from typing import Tuple


class GaussianPolicy(nn.Module):
    action_dim: int
    action_range: Tuple[int, int]
    model: nn.Module

    def setup(self):
        self.action_log_std = self.param(
            "action_log_std", constant(0.0), (self.action_dim,)
        )

    def _action_dist(self, obs):
        action_mean, new_memory= self.model(obs)
        return distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(self.action_log_std)
        ), new_memory

    def __call__(self, obs, rng):
        action_dist, new_memory = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return (action, new_memory), action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs, rng):
        action, _, _ = self(obs, rng)
        action, new_memory = action
        return jnp.clip(action, self.action_range[0], self.action_range[1]), new_memory

    def log_prob_entropy(self, obs, action):
        action_dist, new_memory = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs, rng):
        action_dist, new_memory = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, new_memory, action_dist.log_prob(action)