import abc
import logging
from typing import List

import pyro
import torch
from pyro.distributions import MultivariateNormal, Uniform

from rewards.constants import EPS, NUM_REWARD_FEATS

class AbstractListener(abc.ABC):
    def reset(self):
        """Reset posterior when starting a new game."""
        raise NotImplementedError()

    def observe(self, utterance: str, options: List):
        """Updates posterior with observed utterance and options."""
        raise NotImplementedError()

    def sample_posterior(self, num_samples=1000):
        """Draws reward samples from the posterior given observations so far."""
        raise NotImplementedError


class PosteriorListener(AbstractListener):
    def __init__(self, reward_sample_type="uniform", prior="uniform"):
        assert reward_sample_type in ["uniform", "quantized"]
        self.reward_sample_type = reward_sample_type
        assert prior in ("uniform", "gaussian")
        self.prior = prior
        logging.info(f"Prior: {self.prior}")
        self.reset()

    def reset(self):
        self.emp_marginal = None
        self.observations = []

    def observe(self, utterance: str, options: List, optimal_index=None):
        self.observations.append((utterance, options, optimal_index))

    def sample_posterior(self, num_samples=1000):
        assert (
            self.emp_marginal is not None
        ), "sampling from prior not implemented, check that emp_marginal has been instantiated (e.g. may need at least one obs)"
        samples = (
            torch.stack([self.emp_marginal() for _ in range(num_samples)])
            .squeeze()
            .cpu()
            .numpy()
        )
        return samples

    def _reward_prior(self):
        if self.prior == "uniform":
            r_sampled_list = []
            for i in range(NUM_REWARD_FEATS):
                ri_prior = Uniform(-1.0, 1.0 + EPS)
                ri = pyro.sample(f"r_{i}", ri_prior)
                if self.reward_sample_type == "quantized":
                    # quantize into values [-1.0, -0.5, 0.0, 0.5, 1.0], with equal expected counts for each value
                    ri = (ri * 2.5).round() / 2.0
                r_sampled_list.append(ri)
            return r_sampled_list
        elif self.prior == "gaussian":
            r_prior = MultivariateNormal(
                torch.zeros(NUM_REWARD_FEATS),
                0.5 * torch.eye(NUM_REWARD_FEATS, NUM_REWARD_FEATS),
            )
            r = pyro.sample("r", r_prior)
            return r.tolist()
        else:
            raise NotImplementedError
