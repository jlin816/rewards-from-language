import numpy as np
import torch

from rewards.constants import NUM_REWARD_FEATS
from rewards.utils import generate_options_arr

def choose_option_with_reward_function(options, reward_weights, return_rewards=True):
    if isinstance(options, torch.Tensor):
        options = options.numpy()
    if isinstance(reward_weights, torch.Tensor):
        reward_weights = reward_weights.numpy()
    if isinstance(reward_weights, list):
        reward_weights = np.array(reward_weights)
    assert reward_weights.shape == (NUM_REWARD_FEATS,)
    rewards = np.sum(np.multiply(options, reward_weights), axis=-1)
    opt_index = np.argmax(rewards)
    if return_rewards:
        return opt_index, rewards
    else:
        return opt_index


def calc_regret_and_acc(pred_reward_wts, true_reward_wts, options):
    pred_index = choose_option_with_reward_function(
        options, pred_reward_wts, return_rewards=False
    )
    true_index, true_rewards = choose_option_with_reward_function(
        options, true_reward_wts, return_rewards=True
    )
    assert true_rewards[true_index] == np.max(true_rewards)
    regret = true_rewards[true_index] - true_rewards[pred_index]
    correct = int(true_index == pred_index)
    return regret, correct


def sim_regret(pred_reward_wts, true_reward_wts, eval_sets=None, num_eval_sets=50):
    """Calculate regret and acc on simulated (generated) option sets.

    Args:
        pred_reward_wts
        true_reward_wts
        eval_sets (list of np arrays): feed in to evaluate on specific option sets, or None if they should be randomly generated
        num_eval_sets: number of option sets to eval on, if eval_sets = None
    """
    regrets = []
    corrects = []
    if eval_sets is not None:
        num_eval_sets = len(eval_sets)
    for i in range(num_eval_sets):
        if eval_sets is not None:
            options = eval_sets[i]
        else:
            options = generate_options_arr()
        regret, correct = calc_regret_and_acc(pred_reward_wts, true_reward_wts, options)
        regrets.append(regret)
        corrects.append(correct)
    acc = np.mean(corrects)
    avg_regret = np.mean(regrets)
    return avg_regret, acc
