import einops
import pyro
import torch
from torch import nn
from transformers import BertModel

from rewards.constants import BERT_PAD_ID, NUM_REWARD_FEATS, BERT_BASE_HS
from rewards.models.encoders import (
    FeatureNameEmbeddings,
    LocalizedOptionEncoder,
    LocalizedRewardEncoder,
    OptionEncoder,
    RewardEncoder,
)


class LiteralEmbeddingSpeaker(nn.Module):
    def __init__(
        self,
        embed_size=100,
        feat_embed_size=16,
        reward_embed_size=8,
        hidden_size=256,
        num_layers=1,
        dropout_p=0.4,
        vocab_size=550,
        per_feat_max_reward=True,
        feature_extremes=True,
        unique_extremes=False,
        only_first_option=True,
        language_rep="bert-base",
        latent_type=None,
        latent_reward_weight_priors="uniform_constant",
        latent_reward_weights=[0.0, 1.0],
        language_pooling="max",
        no_mdp_attention=False,
        dot_product_scorer=False,
    ):
        super().__init__()
        self.feat_embed_size = feat_embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.only_first_option = only_first_option
        self.per_feat_max_reward = per_feat_max_reward
        self.feature_extremes = feature_extremes
        self.unique_extremes = unique_extremes

        num_option_feats = NUM_REWARD_FEATS + 1
        if per_feat_max_reward:
            num_option_feats += NUM_REWARD_FEATS
        if feature_extremes:
            num_option_feats += NUM_REWARD_FEATS * 2

        self.language_rep = language_rep
        self.language_pooling = language_pooling
        self.latent_type = latent_type

        feature_name_embedding_size = 32

        self._no_mdp_attention = no_mdp_attention
        self._dot_product_scorer = dot_product_scorer

        # Description encoder
        if language_rep == "bert-base":
            self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.bert_fc = nn.Linear(BERT_BASE_HS, hidden_size)
            self.text_rep_size = hidden_size
        else:
            raise NotImplementedError(f"language_rep {language_rep}")

        self.num_encoded_options = 1
        if self.no_mdp_attention:
            self.option_encoder = OptionEncoder(
                feat_embed_size=feat_embed_size,
                dropout_p=0.0,
                num_feats=num_option_feats,
            )
            self.reward_encoder = RewardEncoder(
                reward_embed_size=reward_embed_size,
                dropout_p=0.0,
            )
            self.option_feature_name_embeddings = None
            self.reward_feature_name_embeddings = None
            self.option_attention_weight = None
            self.reward_attention_weight = None
        else:
            self.option_feature_name_embeddings = (
                option_feature_name_embeddings
            ) = FeatureNameEmbeddings(name_embedding_size=feature_name_embedding_size)
            self.reward_feature_name_embeddings = (
                reward_feature_name_embeddings
            ) = FeatureNameEmbeddings(name_embedding_size=feature_name_embedding_size)
            self.option_encoder = LocalizedOptionEncoder(
                option_feature_name_embeddings,
                indicator_embedding_size=32,
                hidden_size=128,
                dropout_p=dropout_p,
                per_feat_output_size=feat_embed_size,
                per_feat_max_reward=per_feat_max_reward,
                feature_extremes=feature_extremes,
            )
            self.reward_encoder = LocalizedRewardEncoder(
                reward_feature_name_embeddings,
                hidden_size=128,
                dropout_p=dropout_p,
                per_feat_output_size=reward_embed_size,
            )
            attention_key_size = feature_name_embedding_size
            self.option_attention_weight = nn.Sequential(
                *[
                    nn.Linear(attention_key_size + self.text_rep_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_size, 1),
                ]
            )
            self.reward_attention_weight = nn.Sequential(
                *[
                    nn.Linear(attention_key_size + self.text_rep_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_size, 1),
                ]
            )

        if self.dot_product_scorer:
            self.option_nn = nn.Sequential(
                *[
                    nn.Linear(feat_embed_size, hidden_size),
                    nn.Dropout(dropout_p),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                ]
            )
            self.reward_nn = nn.Sequential(
                *[
                    nn.Linear(reward_embed_size, hidden_size),
                    nn.Dropout(dropout_p),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                ]
            )

        # combined encoding
        if self.latent_type is None:
            if not self.dot_product_scorer:
                self.scorer = nn.Sequential(
                    *[
                        ## hierarchical attention
                        nn.Linear(
                            self.num_encoded_options * feat_embed_size
                            + reward_embed_size
                            + self.text_rep_size,
                            hidden_size,
                        ),
                        nn.Dropout(dropout_p),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    ]
                )
        elif self.latent_type == "fuse_scores":
            if not self.dot_product_scorer:
                self.reward_scorer = nn.Sequential(
                    *[
                        nn.Linear(reward_embed_size + self.text_rep_size, hidden_size),
                        nn.Dropout(dropout_p),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    ]
                )
                self.option_scorer = nn.Sequential(
                    *[
                        nn.Linear(
                            self.num_encoded_options * feat_embed_size
                            + self.text_rep_size,
                            hidden_size,
                        ),
                        nn.Dropout(dropout_p),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    ]
                )
            self.latent_reward_weights = nn.Parameter(
                torch.Tensor(latent_reward_weights).float(), requires_grad=False
            )
            assert self.latent_reward_weights.dim() == 1
            assert torch.all(
                (0.0 <= self.latent_reward_weights)
                & (self.latent_reward_weights <= 1.0)
            )
            if latent_reward_weight_priors in ["learned", "uniform_constant"]:
                self.latent_reward_prior_logits = nn.Parameter(
                    torch.zeros_like(self.latent_reward_weights),
                    requires_grad=latent_reward_weight_priors == "learned",
                )
            else:
                p = torch.tensor(latent_reward_weight_priors)
                assert p.size() == self.latent_reward_weights.size()
                self.latent_reward_prior_logits = nn.Parameter(
                    p.log(), requires_grad=False
                )
        else:
            raise NotImplementedError(f"latent_type {self.latent_type}")

        model_config = locals()

        if not self.no_mdp_attention:
            del model_config["option_feature_name_embeddings"]
            del model_config["reward_feature_name_embeddings"]
        # serialize nn.module
        model_config["self"] = str(model_config["self"])
        model_config["__class__"] = str(model_config["__class__"])
        self.model_config = model_config

    @property
    def no_mdp_attention(self):
        # for backward compatibility with serialized models that didn't have this attribute
        # TODO: retrain models without this
        try:
            return self._no_mdp_attention
        except:
            self._no_mdp_attention = False
            return False

    @property
    def dot_product_scorer(self):
        # for backward compatibility with serialized models that didn't have this attribute
        # TODO: retrain models without this
        try:
            return self._dot_product_scorer
        except:
            self._dot_product_scorer = False
            return False

    def encode_text(self, text):
        if self.language_rep == "bert-base":
            attention_mask = text != BERT_PAD_ID
            outs = self.bert_encoder(input_ids=text, attention_mask=attention_mask)
            text_encoded = (
                outs.pooler_output
            )  # gets hs for first [CLS] token, with tanh applied
            text_encoded = self.bert_fc(text_encoded)
        else:
            raise NotImplementedError(f"language_rep {self.language_rep}")
        return text_encoded

    def _attend(
        self,
        encoded_features,
        text_encoded,
        attention_module,
        feature_name_embeddings,
        tile_along_negatives,
    ):
        # return options_encoded

        if self.no_mdp_attention:
            # add bogus feature dimension
            encoded_features = encoded_features.unsqueeze(1)

        if tile_along_negatives:
            assert text_encoded.dim() == 3
            num_text = text_encoded.size(1)
            encoded_features = einops.repeat(
                encoded_features,
                "batch_size num_features d -> batch_size num_text num_features d",
                num_text=num_text,
            )
        else:
            assert text_encoded.dim() == 2
            num_text = 1
            text_encoded = text_encoded.unsqueeze(1)
            encoded_features = encoded_features.unsqueeze(1)

        if self.no_mdp_attention:
            assert encoded_features.size(-2) == 1
            encoded_features = encoded_features.squeeze(-2)
            if not tile_along_negatives:
                assert num_text == 1 and encoded_features.size(1) == 1
                encoded_features = encoded_features.squeeze(1)
            return encoded_features, None

        batch_size = text_encoded.size(0)

        # attention_keys = encoded_features.contiguous()
        attention_keys = einops.repeat(
            feature_name_embeddings(batch_size),
            "batch_size num_features d -> batch_size num_text num_features d",
            num_text=num_text,
        )

        # (batch_size, num_text, d)
        feature_weights = (
            attention_module(
                torch.cat(
                    (
                        einops.repeat(
                            text_encoded,
                            "b num_text d -> b num_text num_features d",
                            num_features=NUM_REWARD_FEATS,
                        ),
                        attention_keys,
                    ),
                    dim=-1,
                )
                # einops.repeat(
                #     text_encoded, "b num_text d -> b num_text num_features d",
                #     num_features=NUM_REWARD_FEATS
                # ),
                # encoded_features.contiguous()
            )
            .squeeze(-1)
            .softmax(-1)
        )
        attended = torch.einsum("btfd,btf->btd", encoded_features, feature_weights)
        if not tile_along_negatives:
            assert num_text == 1 and attended.size(1) == 1
            attended = attended.squeeze(1)
        return attended, feature_weights

    def encode_options(self, options, text_encoded, tile_along_negatives=False):
        # text_encoded: (batch, num_directions(2) * hidden_size)
        # (batch_size, num_options, num_features, d)
        options_encoded = self.option_encoder(options)
        # take only the first option
        assert (
            self.num_encoded_options == 1 and self.only_first_option
        ), "embedding speaker only encoding first option"
        options_encoded = options_encoded[:, 0]
        return self._attend(
            options_encoded,
            text_encoded,
            self.option_attention_weight,
            self.option_feature_name_embeddings,
            tile_along_negatives=tile_along_negatives,
        )

    def encode_reward(self, reward_weights, text_encoded, tile_along_negatives=False):
        # (batch, options_dim)
        reward_encoded = self.reward_encoder(reward_weights)
        return self._attend(
            reward_encoded,
            text_encoded,
            self.reward_attention_weight,
            self.reward_feature_name_embeddings,
            tile_along_negatives=tile_along_negatives,
        )

    def forward(
        self,
        options,
        reward_weights,
        text=None,
        text_encoded=None,
        tile_along_negatives=False,
    ):
        assert text is not None or text_encoded is not None
        if text_encoded is None:
            text_encoded = self.encode_text(text)
        else:
            if text is not None:
                assert text_encoded.size(0) == text.size(1)
            assert text_encoded.size(0) == options.size(0)
        options_encoded, option_attention = self.encode_options(
            options, text_encoded, tile_along_negatives=tile_along_negatives
        )
        reward_encoded, reward_attention = self.encode_reward(
            reward_weights, text_encoded, tile_along_negatives=tile_along_negatives
        )

        extras = {
            "option_attention": option_attention,
            "reward_attention": reward_attention,
        }

        if self.dot_product_scorer:
            options_encoded = self.option_nn(options_encoded)
            reward_encoded = self.reward_nn(reward_encoded)

        if not hasattr(self, "latent_type") or self.latent_type is None:
            if self.dot_product_scorer:
                if text_encoded.dim() == 2:
                    scores = torch.einsum(
                        "bd,bd,bd->b", options_encoded, reward_encoded, text_encoded
                    )
                else:
                    scores = torch.einsum(
                        "btd,btd,btd->bt", options_encoded, reward_encoded, text_encoded
                    )
            else:
                inputs = torch.cat((options_encoded, reward_encoded, text_encoded), -1)
                # (batch,)
                scores = self.scorer(inputs).squeeze(-1)
        elif self.latent_type == "fuse_scores":
            if self.dot_product_scorer:
                if text_encoded.dim() == 2:
                    reward_score = torch.einsum(
                        "bd,bd->b", reward_encoded, text_encoded
                    )
                    option_score = torch.einsum(
                        "bd,bd->b", options_encoded, text_encoded
                    )
                else:
                    assert text_encoded.dim() == 3
                    reward_score = torch.einsum(
                        "btd,btd->bt", reward_encoded, text_encoded
                    )
                    option_score = torch.einsum(
                        "btd,btd->bt", options_encoded, text_encoded
                    )
                reward_score = reward_score.unsqueeze(-1)
                option_score = option_score.unsqueeze(-1)
            else:
                # (batch, text_dim, 1) if tile_along_negatives else (batch, 1)
                reward_score = self.reward_scorer(
                    torch.cat((reward_encoded, text_encoded), -1)
                )
                # (batch, text_dim, 1) if tile_along_negatives else (batch, 1)
                option_score = self.option_scorer(
                    torch.cat((options_encoded, text_encoded), -1)
                )
            extras["reward_score"] = reward_score
            extras["option_score"] = option_score
            if tile_along_negatives:
                assert reward_score.dim() == 3
                weights_tiled = einops.repeat(
                    self.latent_reward_weights,
                    "w -> b t w",
                    b=reward_score.size(0),
                    t=reward_score.size(1),
                )
            else:
                weights_tiled = einops.repeat(
                    self.latent_reward_weights, "w -> b w", b=reward_score.size(0)
                )
            # (batch, len(self.latent_reward_weights))
            scores = reward_score.expand_as(
                weights_tiled
            ) * weights_tiled + option_score.expand_as(weights_tiled) * (
                1.0 - weights_tiled
            )
        else:
            raise NotImplementedError

        return scores, extras

    def add_negatives_and_score_batch(
        self, options, reward_weights, text, hard_negatives
    ):
        batch_size = options.size(0)

        assert self.language_rep == "bert-base"
        num_hard_negatives = hard_negatives.shape[1]
        hard_negatives_reshape = einops.rearrange(
            hard_negatives,
            "batch_size num_hard_negatives seq_len -> (batch_size num_hard_negatives) seq_len ",
        )
        num_text = batch_size + num_hard_negatives
        hard_negatives_encoded = einops.rearrange(
            self.encode_text(hard_negatives_reshape),
            "(batch_size num_hard_negatives) dt -> batch_size num_hard_negatives dt",
            batch_size=batch_size,
            num_hard_negatives=num_hard_negatives,
        )

        text_encoded = self.encode_text(text)
        return self._contrastive_helper(
            options,
            reward_weights,
            text_encoded,
            hard_negatives_encoded,
            num_text,
            include_batch_in_distractors=True,
        )

    def score_contrastive(
        self,
        options,
        reward_weights,
        text=None,
        contrastive=None,
        contrastive_encoded=None,
        text_encoded=None,
    ):
        assert contrastive is not None or contrastive_encoded is not None
        assert text is not None or text_encoded is not None
        batch_size = options.size(0)
        assert reward_weights.size(0) == batch_size
        if text_encoded is None:
            assert text is not None
            assert text.size(1) == batch_size
            text_encoded = self.encode_text(text)
        if contrastive_encoded is None:
            assert contrastive.dim() == 2
            contrastive_encoded = self.encode_text(contrastive)
        else:
            # (num_contrastive, hidden_size*2)
            assert contrastive_encoded.dim() == 2
        num_text = 1 + contrastive_encoded.size(0)
        # (batch_size, d)
        contrastive_encoded = einops.repeat(
            contrastive_encoded,
            "n_distractors dt -> batch_size n_distractors dt",
            batch_size=batch_size,
        )
        return self._contrastive_helper(
            options,
            reward_weights,
            text_encoded,
            contrastive_encoded,
            num_text,
            include_batch_in_distractors=False,
        )

    def _contrastive_helper(
        self,
        options,
        reward_weights,
        text_encoded,
        hard_negatives_encoded,
        num_text,
        include_batch_in_distractors=True,
    ):
        """
        :param options: (batch_size, num_options, num_features)
        :param reward_weights: (batch_size, num_features)
        :param text_encoded:
        :param hard_negatives_encoded:
        :return: logits of size (batch, num_text_utts)
        """
        # (batch_size, num_options, num_features)
        batch_size = options.size(0)
        if include_batch_in_distractors:
            text_encoded_expand = einops.repeat(
                text_encoded,
                "num_text dt -> batch_size num_text dt",
                batch_size=batch_size,
            )
            assert torch.all(text_encoded_expand[0] == text_encoded_expand[1])
            # (batch, num_text, hidden_size)
            text_and_negatives_encoded_expand = torch.cat(
                (text_encoded_expand, hard_negatives_encoded), dim=1
            )
            assert text_and_negatives_encoded_expand.size(0) == batch_size
            assert text_and_negatives_encoded_expand.size(1) == num_text
            assert torch.all(
                text_and_negatives_encoded_expand[:, batch_size]
                == hard_negatives_encoded[:, 0]
            )
            assert torch.all(
                text_and_negatives_encoded_expand[:, batch_size + 1]
                == hard_negatives_encoded[:, 1]
            )
        else:
            text_encoded_expand = text_encoded.unsqueeze(1)
            text_and_negatives_encoded_expand = torch.cat(
                (text_encoded_expand, hard_negatives_encoded), dim=1
            )

        # options_encoded, option_attention = self.encode_options(options, text_and_negatives_encoded_expand, tile_along_negatives=True)
        # reward_encoded, reward_attention = self.encode_reward(reward_weights, text_encoded)
        # options_encoded_expand = einops.repeat(options_encoded, "bo do -> bo bt do", bt=num_text)
        # reward_encoded_expand = einops.repeat(reward_encoded, "bo dr -> bo bt dr", bt=num_text)
        # inputs = torch.cat((options_encoded_expand, reward_encoded_expand, text_and_negatives_encoded_expand), -1)
        # logits = self.scorer(inputs).squeeze(-1)
        # assert logits.size() == (batch_size, num_text)

        return self.forward(
            options,
            reward_weights,
            text_encoded=text_and_negatives_encoded_expand,
            tile_along_negatives=True,
        )

    def joint_with_latents(
        self, utterance_given_weight_logits, normalize_over_utterances=True
    ):
        """
        f(utt, latent, reward, options) -> p(utt, latent | reward, options)
        """
        assert utterance_given_weight_logits.dim() == 3
        # (batch, num_text, len(self.latent_reward_weights))
        if normalize_over_utterances:
            per_weight_log_probs = utterance_given_weight_logits.log_softmax(1)
        else:
            per_weight_log_probs = utterance_given_weight_logits
        # (len(self.latent_reward_weights),)
        weight_log_probs = self.latent_reward_prior_logits.log_softmax(-1)
        # (batch, num_text, len(self.latent_reward_weights))
        joint_log_probs = (
            weight_log_probs.unsqueeze(0).unsqueeze(1).expand_as(per_weight_log_probs)
        ) + per_weight_log_probs
        return joint_log_probs

    def latent_posterior(self, utterance_given_weight_logits):
        """
        f(utt, latent, reward, options) -> p(latent | utt, reward, options)
        """
        joint_log_probs = self.joint_with_latents(utterance_given_weight_logits)
        return joint_log_probs - joint_log_probs.logsumexp(-1, keepdim=True)

    def marginalize_latents(self, logits):
        """
        f(utt, latent, reward, options) -> p(utt | reward, options)
        Return a tensor of (batch_size, |utterance support|), which is normalized (in log-space) over the |utterance support| dimension
        If this is a latent speaker, will marginalize over possible latent weights, using the weight probabilities implicitly given by
        :param logits: the tensor of unnormalized logits returned by _contrastive_helper.
            if self.latent_type is None, must have dimensions (batch_size, |utterance support|)
            if self.latent_type == 'fuse_scores', must have dimensions (batch_size, |utterance support|, len(self.latent_reward_weights))
        TODO: update this to handle tile_along_negative=False?
        """
        if not hasattr(self, "latent_type") or self.latent_type is None:
            assert logits.dim() == 2
            # (batch, batch + num_hard_negs)
            log_probs = logits.log_softmax(dim=-1)
        elif self.latent_type == "fuse_scores":
            assert logits.dim() == 3
            # (batch, batch + num_hard_negs, len(self.latent_reward_weights))
            per_weight_log_probs = logits.log_softmax(1)
            # (len(self.latent_reward_weights),)
            weight_log_probs = self.latent_reward_prior_logits.log_softmax(-1)
            (log_probs,) = pyro.ops.contract.einsum(
                "btw,w->bt",
                per_weight_log_probs,
                weight_log_probs,
                modulo_total=True,
                backend="pyro.ops.einsum.torch_log",
            )
            assert torch.allclose(
                log_probs.exp().sum(-1),
                torch.ones(log_probs.size(0), device=log_probs.device),
            )
        else:
            raise NotImplementedError(f"latent_type {self.latent_type}")
        return log_probs

    def get_reward_only_logits(self, logits):
        if self.latent_type == "fuse_scores":
            weight_list = self.latent_reward_weights.tolist()
            try:
                weight_index = weight_list.index(1.0)
            except ValueError as e:
                print("latent weights must contain 1.0")
                print(e)
                raise e
            return logits[..., weight_index]
        else:
            return logits

    def compute_loss(
        self,
        options,
        reward_weights,
        text,
        hard_negatives,
        infonce=False,
        infonce_latent_posterior_weighting=False,
        infonce_reward_only=False,
    ):
        logits, extras = self.add_negatives_and_score_batch(
            options, reward_weights, text, hard_negatives
        )
        log_probs = self.marginalize_latents(logits)
        assert log_probs.dim() == 2
        # diagonal cuts off hard negatives
        gold_log_probs = torch.diagonal(log_probs, 0)
        loss = -1.0 * gold_log_probs.mean()
        if infonce:
            if logits.dim() == 3:
                # TODO: remove duplicated code from get_reward_only_logits
                weight_list = self.latent_reward_weights.tolist()
                try:
                    weight_index = weight_list.index(1.0)
                except ValueError as e:
                    print("latent weights must contain 1.0")
                    print(e)
                    raise e
                logits_to_use = logits[..., weight_index]
            else:
                assert logits.dim() == 2
                logits_to_use = logits
            listener_log_probs = logits_to_use.log_softmax(dim=0)
            listener_loss_per_instance = -1.0 * torch.diagonal(listener_log_probs, 0)
            if infonce_latent_posterior_weighting:
                assert logits.dim() == 3
                # Get p(reward latent=1.0 | utt, xi, reward) and use it to
                # weight the listener_loss_per_instance.
                # latent_posterior(logits): # (batch_size, num_utts, num_latents)
                #       where the diagonal entries (along dims 0 and 1)
                #       correspond to true matches between (reward_weights,
                #       options) and (utterance).
                # So we should get those diagonal entries (for weight 1.0 in
                # the num_latents dimension).
                latent_posterior_log_p = self.latent_posterior(logits)[
                    ..., weight_index
                ].diagonal(0)
                listener_loss_per_instance *= latent_posterior_log_p.exp().detach()
            if infonce_reward_only:
                loss = listener_loss_per_instance.mean()
            else:
                loss += listener_loss_per_instance.mean()
                loss /= 2
        return loss

    def get_log_probs(
        self,
        reward_weights,
        options,
        utterances=None,
        utterances_encoded=None,
        distractors=None,
        distractors_encoded=None,
        fixed_reward_weight=None,
    ):
        """
        Batched computation of log p(utt[i] | reward_weights[i], options[i]),
        where the normalizer is computed as \sum_{j} p(distractors[j] | reward_weights[i], options[i])
        (i.e. the same distractors are used across all instances in the batch)
        Inputs are tensors.
        :param reward_weights: tensor (batch_size, NUM_REWARD_FEATURES)
        :param options:  tensor (batch_size, NUM_OPTIONS, num_features)
        :param utterances: optional (either it or utterances encoded must be passed) tensor TODO: dimensions
        :param utterances_encoded: optional (either it or utterances encoded must be passed) tensor (batch_size, text_hidden_size)
        :param distractors: optional (either it or utterances must be passed) tensor TODO: dimensions
        :param distractors_encoded: optional (either it or utterances must be passed) tensor (num_distractors, text_hidden_size)
        :return: log probability tensor of size (batch_size,)
        """
        logits, extras = self.score_contrastive(
            options,
            reward_weights,
            text=utterances,
            text_encoded=utterances_encoded,
            contrastive=distractors,
            contrastive_encoded=distractors_encoded,
        )
        if fixed_reward_weight is not None:
            assert self.latent_type is not None
            weight_index = self.latent_reward_weights.tolist().index(
                fixed_reward_weight
            )
            joint_log_probs = self.joint_with_latents(logits)
            log_probs = joint_log_probs[..., weight_index]
        else:
            log_probs = self.marginalize_latents(logits)
        return log_probs[:, 0]
