import flax.linen as nn


class Critic(nn.Module):
    model: nn.Module

    @nn.compact
    def __call__(self, obs):
        x, _ = self.model(obs)
        return nn.Dense(1)(x).squeeze()
