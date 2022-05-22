import json
import random
import subprocess
import sys
from collections import namedtuple

import numpy as np
import torch

from rewards.constants import (
    NUM_OPTIONS,
    NUM_REWARD_FEATS,
    OPTION_FEATURES,
    REWARD_KEYS,
    FeatureType,
)

ExampleTensor = namedtuple("ExampleTensor", ["utt", "reward_weights", "options"])


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_config(model, path):
    with open(f"{path}_config.json", "w") as f:
        json.dump(model.model_config, f, indent=4)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


# **Note: S0 can see the additional indicator features on each option, but L\* should NOT.**
# L0 should only condition on the options, and when evaluating S0(L|xi, theta,M), 
# ssemble the indicator features given the candidate xi / theta being considered.
def expand_categorical_var(feat_name, categories, selected_category):
    opt = {f"{feat_name}={cat}": 0 for cat in categories}
    opt[f"{feat_name}={selected_category}"] = 1
    return opt


def extremal_indicators(x, fn=np.max, unique=False):
    extremes = fn(x, axis=0)
    indicators = np.equal(x, extremes).astype(int)
    mask_non_unique = np.all(indicators, axis=0)
    indicators[:, mask_non_unique] = 0
    # sanity check: no feature should have col of all 1s
    assert not np.any(np.all(indicators, axis=0))

    if unique:
        indicators[:, np.sum(indicators, axis=0) > 1] = 0
        # sanity check: no feature should have multiple 1s
        assert not np.any(np.sum(indicators, axis=0) > 1)
    return indicators


def augment_options_for_model(
    options, reward, model, return_type="list", override_optimal_index=None
):
    return augment_options_with_indicators(
        options,
        reward,
        add_per_feat_max_reward=model.per_feat_max_reward,
        add_feature_extremes=model.feature_extremes,
        unique_extremes=model.unique_extremes,
        return_type=return_type,
        override_optimal_index=override_optimal_index,
    )


def augment_options_for_listener_model(options, model, return_type="list", device=None):
    return augment_options_with_indicators_listener(
        options,
        add_feature_extremes=model.indicators["feature_extremes"],
        unique_extremes=model.indicators["unique_extremes"],
        return_type=return_type,
        device=device,
    )


def augment_options_with_indicators(
    options,
    reward,
    add_per_feat_max_reward=True,
    add_feature_extremes=False,
    unique_extremes=False,
    return_type="list",
    override_optimal_index=None,
):
    """
    Args:
        options: list of options, each a list of ints
        reward: list of reward weights
    """
    assert (
        len(options) == NUM_OPTIONS and len(options[0]) == NUM_REWARD_FEATS
    ), f"Wrong options shape {options}"
    assert len(reward) == NUM_REWARD_FEATS

    per_feat_rewards = np.multiply(options, reward)
    if add_per_feat_max_reward:
        option_is_max = extremal_indicators(
            per_feat_rewards, fn=np.max, unique=unique_extremes
        )
        augmented_options = np.concatenate((options, option_is_max), axis=-1)
    else:
        augmented_options = options

    if add_feature_extremes:
        value_is_max = extremal_indicators(options, fn=np.max, unique=unique_extremes)
        value_is_min = extremal_indicators(options, fn=np.min, unique=unique_extremes)
        augmented_options = np.concatenate(
            (augmented_options, value_is_max, value_is_min), axis=-1
        )

    if hasattr(augmented_options, "tolist"):
        augmented_options = augmented_options.tolist()

    # Put best option first for speaker
    if override_optimal_index is not None:
        optimal_index = override_optimal_index
    else:
        optimal_index = np.argmax(np.sum(per_feat_rewards, axis=-1))
    option_is_global_max = np.zeros((3, 1))
    option_is_global_max[0] = 1
    augmented_options.insert(0, augmented_options.pop(optimal_index))

    augmented_options = np.concatenate(
        (augmented_options, option_is_global_max), axis=-1
    )
    if return_type == "list":
        return augmented_options.tolist()
    elif return_type == "numpy":
        return augmented_options
    elif return_type == "torch":
        return torch.as_tensor(augmented_options, dtype=torch.float)
    else:
        raise NotImplementedError(f"return type {return_type}")


def augment_options_with_indicators_listener(
    options,
    add_feature_extremes=True,
    unique_extremes=False,
    return_type="list",
    device=None,
):
    augmented_options = options
    if add_feature_extremes:
        value_is_max = extremal_indicators(options, fn=np.max, unique=unique_extremes)
        value_is_min = extremal_indicators(options, fn=np.min, unique=unique_extremes)
        augmented_options = np.concatenate(
            (augmented_options, value_is_max, value_is_min), axis=-1
        )

    if return_type == "list":
        return augmented_options.tolist()
    elif return_type == "numpy":
        return augmented_options
    elif return_type == "torch":
        return torch.as_tensor(
            augmented_options, dtype=torch.float, device=device if device else "cpu"
        )
    else:
        raise NotImplementedError(f"return type {return_type}")


def deaugment_options(augmented_options):
    if augmented_options.shape[0] == NUM_OPTIONS:
        return augmented_options[:, :NUM_REWARD_FEATS]
    # with batch size
    assert augmented_options.shape[1] == NUM_OPTIONS
    return augmented_options[:, :, :NUM_REWARD_FEATS]


def test_augment_options_with_indicators():
    # Test augment_options_with_indicators
    true_options_feat_only = [
        [0.8, 0.0, 0.0, 1.0, 0.0, 0.56, 1.0, 0.62],
        [0.04, 0.0, 1.0, 0.0, 0.0, 0.64, 0.25, 0.28],
        [0.32, 0.0, 0.0, 1.0, 0.0, 0.84, 0.25, 0.16],
    ]
    true_reward_weights = [0, -1, 0.5, 1, 0, 0.5, 1, -1]
    true_options_with_indicators = [
        [
            0.8,
            0.0,
            0.0,
            1.0,
            0.0,
            0.56,
            1.0,
            0.62,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ],
        [
            0.04,
            0.0,
            1.0,
            0.0,
            0.0,
            0.64,
            0.25,
            0.28,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.32,
            0.0,
            0.0,
            1.0,
            0.0,
            0.84,
            0.25,
            0.16,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ],
    ]
    options_with_feat = augment_options_with_indicators(
        true_options_feat_only, true_reward_weights
    )
    assert options_with_feat == true_options_with_indicators


def generate_options_arr():
    opts = []
    for _ in range(3):
        opt = {}
        for feat_key, feat in OPTION_FEATURES.items():
            if feat["type"] == FeatureType.CATEGORICAL:
                sampled_val = random.choice(feat["values"])
                expanded_val = expand_categorical_var(
                    feat_key, feat["values"], sampled_val
                )
                opt.update(expanded_val)
            elif feat["type"] == FeatureType.NUMERICAL:
                val_min, val_max = feat["values"]
                # val = random.randint(val_min, val_max)
                val = np.random.uniform(val_min, val_max)
                # normalize to [0,1]
                normalized_val = (val - val_min) / (val_max - val_min)
                opt[feat_key] = normalized_val
            else:
                raise "unknown feature type"

        opts.append(opt)
    return np.array([[opt[k] for k in REWARD_KEYS] for opt in opts])


def optimal_index(reward_weights, options):
    reward_weights = np.array(reward_weights)
    options = np.array(options)
    return np.argmax(np.matmul(options, reward_weights))


def batch_optimal_index(reward_weights, options):
    reward_weights = np.array(reward_weights)
    options = np.array(options)
    # (batch, n_feat) x (n_feat, T) (batch, 3)
    rewards = np.matmul(reward_weights, options.T)
    return np.argmax(rewards, axis=-1)


def get_carrier_from_list(flight):
    for i, key in enumerate(REWARD_KEYS):
        if key.startswith("carrier=") and flight[i] != 0.0:
            return key.split("=")[1], i


def dump_git_status(
    out_file=sys.stdout,
    exclude_file_patterns=["*.ipynb", "*.th", "*.sh", "*.txt", "*.json"],
):
    subprocess.call("git rev-parse HEAD", shell=True, stdout=out_file)
    exclude_string = " ".join("':(exclude){}'".format(f) for f in exclude_file_patterns)
    subprocess.call(
        "git --no-pager diff -- . {}".format(exclude_string),
        shell=True,
        stdout=out_file,
    )


def all_equal(xs):
    xs = list(xs)
    return all(x == xs[0] for x in xs[1:])
