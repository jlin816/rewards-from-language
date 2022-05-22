import logging

import einops
import pyro
import torch
import tqdm
from pyro.infer import EmpiricalMarginal, Importance
from transformers import BertTokenizer

from rewards.data import FlightTaskDataset, collate_fn_bert, create_dataloaders
from rewards.constants import REPO_PATH
from rewards.models import LiteralEmbeddingSpeaker
from rewards.models.listener import BertEncoderOptionListener
from rewards.posterior_models.posterior_listener import PosteriorListener
from rewards.utils import (
    all_equal,
    augment_options_for_listener_model,
    augment_options_for_model,
    deaugment_options,
)



class L0PragmaticListener(PosteriorListener):
    """A pragmatic listener based on trained listener models p_L0(theta | u, M)."""

    def __init__(
        self, l0_models, normalizers_type="train-filtered", num_samples=1000, cuda=True
    ):
        super().__init__()
        self.l0_models = l0_models
        self.tok = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cuda = cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.num_samples = num_samples

        for model in self.l0_models:
            model.eval()
            if self.cuda:
                model.cuda()

        for model in self.l0_models:
            assert (
                model.indicators == self.l0_models[0].indicators
            ), "obs options are collated assuming all listeners use \
                    the same indicators"
            assert isinstance(
                model, BertEncoderOptionListener
            ), "assumes bert model, tokenizer is hard-coded"

        self.utt_normalizers_path = f"{REPO_PATH}/data/train.jsonl"
        self.utt_normalizers = self._get_normalizers(normalizers_type)

    def _get_normalizers(self, normalizers_type, bert_encoding=True):
        """Get utterance normalizers from the training set."""
        assert normalizers_type == "train-filtered" or normalizers_type == "train-all"
        logging.info("Using normalizer type %s", normalizers_type)
        # Indicators don't matter, we'll just use the utterances.
        ds_args = dict(
            add_feature_extremes=True, unique_extremes=False, for_listener=True
        )
        train_ds = FlightTaskDataset(self.utt_normalizers_path, **ds_args)
        train_dl = create_dataloaders(
            train_ds,
            1.0,
            val_split=0.0,
            train_batch_size=2500,
            collate_fn=collate_fn_bert,
        )["train"]
        utt_normalizers = next(iter(train_dl))["utterance"]
        utt_normalizers = torch.unique(utt_normalizers, dim=0)

        if normalizers_type == "train-filtered" or not bert_encoding:
            all_utterances = self.tok.batch_decode(
                utt_normalizers, skip_special_tokens=True
            )
            if normalizers_type == "train-filtered":
                _utts_to_encode = set(
                    utt
                    for utt in all_utterances
                    if len(utt.split()) <= 8 and len(set(utt) & set("1234567890")) == 0
                )
            else:
                _utts_to_encode = all_utterances
            _utts_to_encode = list(sorted(_utts_to_encode))
            if bert_encoding:
                utt_normalizers = self.tok(
                    _utts_to_encode, return_tensors="pt", padding=True
                )["input_ids"]

        if self.cuda:
            utt_normalizers = utt_normalizers.cuda()

        return utt_normalizers

    def _collate_example(self, utt, opt):
        utterance = self.tok.encode(utt, return_tensors="pt")
        options = augment_options_for_listener_model(
            opt, self.l0_models[0], return_type="torch"
        )
        if self.cuda:
            utterance = utterance.cuda()
            options = options.cuda()
        return utterance, options

    def _log_p_s1(self, utt, rw, options, cache, return_distribution=False):
        """Calculates normalized log p(utt | rw, options)."""
        raise NotImplementedError()

    def _l2_model(self, utt, options, cache, *s1_args, pbar=None, **s1_kwargs):
        """Define the pragmatic model p_l2(theta | utt, options)."""
        if self.emp_marginal is not None:
            r_sampled = self.emp_marginal()  # draw from current posterior
        else:
            r_sampled = torch.tensor(self._reward_prior(), dtype=torch.float)
        if self.cuda:
            r_sampled = r_sampled.cuda()
        lp = self._log_p_s1(utt, r_sampled, options, cache, *s1_args, **s1_kwargs)
        pyro.factor("utt", lp)
        if pbar is not None:
            pbar.update(1)
        return r_sampled

    def observe(self, utterance, options):
        super().observe(utterance, options)
        print(f"Updating with utterance")
        self._update_posterior()

    def _update_posterior(self):
        """Re-sample from the posterior with the current set of observations."""
        with tqdm.tqdm(total=self.num_samples * 2, desc="sampling", ncols=80) as pbar:
            with torch.no_grad():
                importance = Importance(
                    self._l2_model, guide=None, num_samples=self.num_samples
                )
                last_utt, last_opt, _ = self.observations[-1]
                last_obs_utt, last_obs_option = self._collate_example(
                    last_utt, last_opt
                )
                cache = self._cache_precomputed_option_probs(
                    last_obs_utt, last_obs_option
                )
                x = importance.run(last_obs_utt, last_obs_option, cache, pbar=pbar)
        self.emp_marginal = EmpiricalMarginal(x)

    def _log_p_s1_single(self, utt, rw, options, return_distribution=False):
        """Gets log p_s1(utt | rw, options) for a single example, without optimizations for sampling."""
        assert len(options) == 3 and len(options[0]) == 8 and len(rw) == 8
        utt, options = self._collate_example(utt, options)
        rw = torch.tensor(rw, dtype=torch.float, device=self.device)
        cache = self._cache_precomputed_option_probs(utt, options)
        return self._log_p_s1(
            utt, rw, options, cache, return_distribution=return_distribution
        )


class L0AndS0PragmaticListener(L0PragmaticListener):
    """A pragmatic listener based on trained listener models p_L0(theta | u, M) and
    trained speaker models p_S0(u | theta, M)."""

    def __init__(
        self,
        l0_models,
        s0_models,
        normalizers_type="train-filtered",
        nearsightedness_lambda=None,
        speaker_beta=1.0,
        s1_variant="log-interpolate",
        normalize_per_term=False,
        nearsighted_temperature=None,
        farsighted_temperature=None,
        cuda=False,
    ):
        super().__init__(l0_models, normalizers_type=normalizers_type, cuda=cuda)

        assert all_equal(model.language_rep for model in s0_models)

        self.s0_bert_encoding = True

        self.s0_utt_normalizers = self._get_normalizers(
            normalizers_type, bert_encoding=self.s0_bert_encoding
        )

        if self.s0_bert_encoding:
            assert self.s0_utt_normalizers.size(0) == self.utt_normalizers.size(0)

        self.s0_models = s0_models
        for model in self.s0_models:
            model.eval()
            if self.cuda:
                model.cuda()

        for model in self.s0_models:
            assert isinstance(
                model, LiteralEmbeddingSpeaker
            ), "assumes bert model, tokenizer is hard-coded"
            assert model.feature_extremes == self.s0_models[0].feature_extremes
            assert model.unique_extremes == self.s0_models[0].unique_extremes

        self.speaker_beta = speaker_beta
        assert s1_variant in ("interpolate", "log-interpolate")
        self.s1_variant = s1_variant
        self.normalize_per_term = normalize_per_term
        if self.s1_variant == "log-interpolate" or self.s1_variant == "interpolate":
            assert 0 <= nearsightedness_lambda <= 1
            self.nearsightedness_lambda = nearsightedness_lambda
        self.speaker_fixed_reward_weight = 1.0

        self.farsighted_temperature = farsighted_temperature
        self.nearsighted_temperature = nearsighted_temperature
        if self.farsighted_temperature is not None:
            assert (
                self.normalize_per_term
            ), "temp is only applied right before softmax normalization"

    def _collate_example(self, utt, opt):
        listener_utterance, listener_options = super()._collate_example(utt, opt)
        speaker_utterance = listener_utterance
        return (listener_utterance, speaker_utterance), listener_options

    def _cache_precomputed_option_probs(self, listener_and_speaker_utts, options):
        """Precompute p_l0(o_i | utt, options) since it's not reward-dependent.
        listener_and_speaker_utts = (listener_utt, speaker_utt): both represent the
        same string utterance, but may differ in representation
        """
        listener_utt, speaker_utt = listener_and_speaker_utts

        # list of len `num_models`, each (num_utts, num_options)
        pl0_by_model_obs_mdp = []
        for model_ix, listener in enumerate(tqdm.tqdm(self.l0_models)):
            with torch.no_grad():
                normalizers_encoded = listener.encode_utt(self.utt_normalizers)
                listener_utterance_encoded = listener.encode_utt(listener_utt)

            # Get p_l0(xi | u, M) in observed mdp M
            opt_batch = einops.repeat(
                options, "o f -> b o f", b=len(self.utt_normalizers)
            )
            with torch.no_grad():
                normalizer_probs = listener(
                    None, opt_batch, encoded_utt=normalizers_encoded
                )
                listener_this_utt_probs = listener.forward(
                    None, options.unsqueeze(0), encoded_utt=listener_utterance_encoded
                )
            pl0_by_model_obs_mdp.append(
                torch.cat((normalizer_probs, listener_this_utt_probs), dim=0).softmax(
                    -1
                )
            )

        # Ensemble model probabilities
        # p_{L0}(o | u, m)
        # (num_utterances, num_options) after ensembling
        pl0_opt_given_utt_obs_mdp = torch.stack(pl0_by_model_obs_mdp, dim=0)
        pl0_opt_given_utt_obs_mdp = pl0_opt_given_utt_obs_mdp.mean(0)
        norm_check = pl0_opt_given_utt_obs_mdp.sum(-1)
        assert torch.allclose(norm_check, torch.ones_like(norm_check))
        with torch.no_grad():
            speaker_normalizers_encoded = [
                model.encode_text(self.s0_utt_normalizers) for model in self.s0_models
            ]

        return pl0_opt_given_utt_obs_mdp, speaker_normalizers_encoded

    def _log_p_s1_near(self, listener_and_speaker_utt, rw, opts, cache):
        """Calculates p_action(utt | rw) with a base listener model L0. 
        Returns:
           p(utt | xi*_theta in this mdp [opts]): shape (|utt_normalizers| + 1,)
        """
        pl0_opt_given_utt_obs_mdp, _ = cache
        optimal_idx_obs_mdp = torch.einsum(
            "of,f->o", deaugment_options(opts), rw
        ).argmax(-1)
        pl0_correct_opt_given_utt_obs_mdp = pl0_opt_given_utt_obs_mdp[
            :, optimal_idx_obs_mdp
        ]
        return pl0_correct_opt_given_utt_obs_mdp

    def _log_p_s1_far(self, listener_and_speaker_utt, rw, opts, cache):
        """Calculates p_reward(utt | rw) by normalizing s0 embedding speaker scores over u.
        Returns:
           p(utt | rw): shape (|utt_normalizers| + 1,)
        """
        _, speaker_utt = listener_and_speaker_utt
        _, speaker_normalizers_encoded = cache

        # add batch dimension
        reward_weights = rw.unsqueeze(0)
        options = opts.unsqueeze(0)
        assert reward_weights.dim() == 2
        assert options.dim() == 3
        deaug_options = deaugment_options(options.squeeze(0))

        speaker_options = []
        for rw in reward_weights:
            speaker_options.append(
                augment_options_for_model(
                    deaug_options.cpu().numpy(),
                    rw.cpu().numpy(),
                    self.s0_models[0],
                    return_type="list",
                )
            )

        speaker_options = torch.tensor(speaker_options).to(options.device)

        assert len(self.s0_models) == len(speaker_normalizers_encoded)
        all_utt_log_probs = []
        with torch.no_grad():
            for model, normalizers_encoded in zip(
                self.s0_models, speaker_normalizers_encoded
            ):
                assert speaker_utt.dim() == 2
                utt_encoded = model.encode_text(speaker_utt)
                # (|utt_normalizers|+1, encoded_text_dim)
                text_encoded = torch.cat((normalizers_encoded, utt_encoded), dim=0)
                # add batch_dim, (1, |utt_normalizers|+1, encoded_text_dim)
                text_encoded = text_encoded.unsqueeze(0)
                # (1, |utt_normalizers| + 1, len(model.latent_reward_weights))
                utt_scores, _ = model.forward(
                    speaker_options,
                    reward_weights,
                    text_encoded=text_encoded,
                    tile_along_negatives=True,
                )
                if self.speaker_fixed_reward_weight is not None:
                    reward_index = model.latent_reward_weights.tolist().index(
                        self.speaker_fixed_reward_weight
                    )
                    # (1, |utt_normalizers| + 1)
                    utt_log_probs = model.joint_with_latents(
                        utt_scores, normalize_over_utterances=True
                    )[:, :, reward_index]
                    # need to renormalize in log space because we're missing probability mass
                    utt_log_probs = utt_log_probs.log_softmax(-1)
                else:
                    utt_log_probs = model.marginalize_latents(utt_scores)
                # remove batch dimension
                utt_log_probs = utt_log_probs.squeeze(0)
                all_utt_log_probs.append(utt_log_probs)
            all_utt_log_probs = torch.stack(all_utt_log_probs, dim=0)

            # Ensemble base model probabilities.
            # (num_ensemble_models, |utt_normalizers| + 1) -> (|utt_normalizers| + 1)
            utt_probs = all_utt_log_probs.exp().mean(0)
            norm_check = utt_probs.sum(-1)
            assert torch.allclose(norm_check, torch.ones_like(norm_check))

            utt_log_probs = utt_probs.log()
            assert utt_log_probs.shape == (len(self.utt_normalizers) + 1,)

        return utt_log_probs

    def _log_p_s1(
        self, listener_and_speaker_utt, rw, opts, cache, return_distribution=False
    ):
        pl0_correct_opt_given_utt_obs_mdp = self._log_p_s1_near(
            listener_and_speaker_utt, rw, opts, cache
        )

        utt_log_probs = self._log_p_s1_far(listener_and_speaker_utt, rw, opts, cache)

        if self.normalize_per_term:
            pl0_correct_opt_given_utt_obs_mdp /= pl0_correct_opt_given_utt_obs_mdp.sum(
                0
            )
            if self.farsighted_temperature is not None:
                utt_log_probs = utt_log_probs / self.farsighted_temperature
            utt_log_probs = utt_log_probs.log_softmax(0)

        if self.s1_variant == "log-interpolate":
            # (num_utts,)
            near_lp = pl0_correct_opt_given_utt_obs_mdp.log()
            log_p_s1 = self.speaker_beta * (
                self.nearsightedness_lambda * near_lp
                + (1 - self.nearsightedness_lambda) * utt_log_probs
            )
            assert log_p_s1.shape == (len(self.utt_normalizers) + 1,)
        elif self.s1_variant == "interpolate":
            log_p_s1 = self.speaker_beta * (
                self.nearsightedness_lambda * pl0_correct_opt_given_utt_obs_mdp
                + (1 - self.nearsightedness_lambda) * utt_log_probs.exp()
            )
            # we have interpolated probs, so we need to take the log
            log_p_s1 = log_p_s1.log()
        else:
            raise NotImplementedError()

        log_p_s1 = log_p_s1.log_softmax(0)

        if return_distribution:
            return log_p_s1

        return log_p_s1[-1]
