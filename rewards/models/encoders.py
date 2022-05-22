import einops
import torch
import torch.nn.functional as F
from torch import nn

from rewards.constants import NUM_FEATS, NUM_OPTIONS, NUM_REWARD_FEATS

class OptionEncoder(nn.Module):
    def __init__(self, feat_embed_size=16, dropout_p=0.4, num_feats=NUM_FEATS):
        super().__init__()
        self.feat_embed_size = feat_embed_size
        self.num_feats = num_feats
        self.option_encoder_lin1 = nn.Linear(num_feats, feat_embed_size)
        self.option_encoder_dropout = nn.Dropout(dropout_p)

    def forward(self, options):
        """Encodes each option with the same weights.
        Args:
            options: shape (batch_size, NUM_OPTIONS, self.num_feats)
        Returns:
            shape (batch_size, NUM_OPTIONS, self.feat_embed_size)
        """
        assert (
            options.shape[1] == NUM_OPTIONS and options.shape[2] == self.num_feats
        ), f"Wrong options shape: {options.shape}"
        x = options
        x = self.option_encoder_lin1(x)
        x = F.relu(x)
        x = self.option_encoder_dropout(x)
        return x


class RewardEncoder(nn.Module):
    def __init__(self, reward_embed_size=16, dropout_p=0.4):
        super().__init__()
        self.reward_encoder_lin1 = nn.Linear(NUM_REWARD_FEATS, reward_embed_size)
        self.reward_encoder_dropout = nn.Dropout(dropout_p)

    def forward(self, reward_weights):
        reward_weights = self.reward_encoder_lin1(reward_weights)
        reward_weights = F.relu(reward_weights)
        reward_weights = self.reward_encoder_dropout(reward_weights)
        return reward_weights


class FeatureNameEmbeddings(nn.Module):
    def __init__(self, name_embedding_size=32):
        super().__init__()
        self.name_embedding_size = name_embedding_size
        self.name_embeddings = nn.Embedding(NUM_REWARD_FEATS, name_embedding_size)

    def forward(self, batch_size):
        embeddings = einops.repeat(
            self.name_embeddings.weight,
            "num_reward_feats d -> batch_size num_reward_feats d",
            batch_size=batch_size,
        )
        return embeddings


class LocalizedOptionEncoder(nn.Module):
    MAX_REWARD = 0
    MAX_FEATURE = 1
    MIN_FEATURE = 2
    OPTION_OPTIMAL = 3
    INDICATOR_TYPES = list(range(4))

    def __init__(
        self,
        feature_name_embeddings: FeatureNameEmbeddings,
        indicator_embedding_size=32,
        hidden_size=128,
        dropout_p=0.4,
        per_feat_output_size=64,
        per_feat_max_reward=True,
        feature_extremes=False,
        unique_extremes=False,
    ):
        super().__init__()
        self.feature_name_embeddings = feature_name_embeddings
        self.indicator_embedding_size = indicator_embedding_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.per_feat_output_size = per_feat_output_size

        self.per_feat_max_reward = per_feat_max_reward
        self.feature_extremes = feature_extremes
        self.unique_extremes = unique_extremes

        self.indicator_embeddings = nn.Embedding(
            len(LocalizedOptionEncoder.INDICATOR_TYPES) * 2, indicator_embedding_size
        )

        self.indicators_used = []
        if self.per_feat_max_reward:
            self.indicators_used.append(LocalizedOptionEncoder.MAX_REWARD)
        if self.feature_extremes:
            self.indicators_used.extend(
                [LocalizedOptionEncoder.MAX_FEATURE, LocalizedOptionEncoder.MIN_FEATURE]
            )
        self.per_feat_input_size = (
            feature_name_embeddings.name_embedding_size
            + 1
            + indicator_embedding_size * len(self.indicators_used)
        )

        def construct_input_fc():
            return nn.Sequential(
                *[
                    nn.Linear(self.per_feat_input_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.hidden_size, self.per_feat_output_size),
                ]
            )

        self.input_fcs = nn.ModuleList(
            [construct_input_fc() for _ in range(NUM_REWARD_FEATS)]
        )

    def embed_indicators(self, indicator_type: int, indicator_values: torch.Tensor):
        assert indicator_type in LocalizedOptionEncoder.INDICATOR_TYPES
        assert torch.all((indicator_values == 0) | (indicator_values == 1))
        indicator_values = indicator_values.long()
        embedding_indices = 2 * indicator_type + indicator_values
        return self.indicator_embeddings(embedding_indices)

    def forward(self, options):
        # options: shape (batch_size, NUM_OPTIONS, self.num_feats)
        batch_size, num_options, _ = options.size()
        feature_values = options[..., :NUM_REWARD_FEATS]
        to_concat = [
            einops.repeat(
                self.feature_name_embeddings(batch_size),
                "batch_size num_feats d -> batch_size num_options num_feats d",
                num_options=num_options,
            ),
            feature_values.unsqueeze(-1),
        ]
        k = NUM_REWARD_FEATS
        if self.per_feat_max_reward:
            to_concat.append(
                self.embed_indicators(
                    LocalizedOptionEncoder.MAX_REWARD,
                    options[..., k : k + NUM_REWARD_FEATS],
                )
            )
            k += NUM_REWARD_FEATS
        if self.feature_extremes:
            to_concat.append(
                self.embed_indicators(
                    LocalizedOptionEncoder.MAX_FEATURE,
                    options[..., k : k + NUM_REWARD_FEATS],
                )
            )
            k += NUM_REWARD_FEATS
            to_concat.append(
                self.embed_indicators(
                    LocalizedOptionEncoder.MIN_FEATURE,
                    options[..., k : k + NUM_REWARD_FEATS],
                )
            )
            k += NUM_REWARD_FEATS

        # (batch_size, num_options, num_features, self.per_feat_input_size)
        x = torch.cat(to_concat, dim=-1)
        # (batch_size, num_options, num_features, self.per_feat_output_size)
        xs = [self.input_fcs[i](x[..., i, :]) for i in range(NUM_REWARD_FEATS)]
        x = torch.stack(xs, dim=-2)
        # x = self.input_fc(x)
        return x


class LocalizedRewardEncoder(nn.Module):
    def __init__(
        self,
        feature_name_embeddings: FeatureNameEmbeddings,
        hidden_size=128,
        dropout_p=0.4,
        per_feat_output_size=64,
    ):
        super().__init__()
        self.feature_name_embeddings = feature_name_embeddings
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.per_feat_output_size = per_feat_output_size

        self.per_feat_input_size = feature_name_embeddings.name_embedding_size + 1

        def construct_input_fc():
            return nn.Sequential(
                *[
                    nn.Linear(self.per_feat_input_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.hidden_size, self.per_feat_output_size),
                ]
            )

        self.input_fcs = nn.ModuleList(
            [construct_input_fc() for _ in range(NUM_REWARD_FEATS)]
        )

    def forward(self, reward_weights: torch.Tensor):
        # reward_weights: (batch_size, NUM_REWARD_FEATS)
        batch_size = reward_weights.size(0)
        to_concat = [
            self.feature_name_embeddings(batch_size),
            reward_weights.unsqueeze(-1),
        ]
        x = torch.cat(to_concat, dim=-1)
        # x = self.input_fc(x)
        xs = [self.input_fcs[i](x[..., i, :]) for i in range(NUM_REWARD_FEATS)]
        x = torch.stack(xs, dim=-2)
        return x
