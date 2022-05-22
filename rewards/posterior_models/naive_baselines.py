import random
from pyro.distributions import Uniform
import torch

from rewards.constants import EPS, NUM_REWARD_FEATS
from rewards.posterior_models.posterior_listener import AbstractListener


class OracleListener(AbstractListener):

    def __init__(self, num_uniform_feats=0):
        """
        Args:
            true_reward:
            num_uniform_feats: number of reward feats that should be uniform instead of centered around the true reward
        """
        self.num_uniform_feats = num_uniform_feats

    def reset(self):
        # do nothing
        pass

    def observe_reward(self, true_reward):
        self.true_reward = torch.tensor(true_reward, dtype=torch.float)
        # Sample new features to be uniform for each reward / trial we observe
        self.uniform_feats = random.sample(
            range(NUM_REWARD_FEATS), self.num_uniform_feats
        )

    def sample_posterior(self, num_samples=1000):
        with torch.no_grad():
            unif = Uniform(
                -torch.ones(num_samples, self.num_uniform_feats),
                torch.ones(num_samples, self.num_uniform_feats) + EPS,
            )
            samples = self.true_reward.repeat((num_samples, 1))
            samples[:, self.uniform_feats] = unif()
            return samples.numpy()
