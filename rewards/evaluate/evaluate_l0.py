import json
import logging
import os
import pickle
import pprint
import random
import time
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pyro
import torch
import wandb
from tqdm import tqdm

from rewards.constants import REPO_PATH
from rewards.evaluate.evaluate_utils import sim_regret
from rewards.models.embedding_speaker import LiteralEmbeddingSpeaker
from rewards.models.listener import BertEncoderOptionListener
from rewards.posterior_models.l0_pragmatic_listener import \
    L0AndS0PragmaticListener
from rewards.posterior_models.naive_baselines import OracleListener
from rewards.posterior_models.posterior_listener import AbstractListener
from rewards.utils import dump_git_status, generate_options_arr, load_model


def run_model_on_games(
    model: AbstractListener,
    games: List[Dict],
    held_out_option_sets: np.array,
    save_dir: str,
    debug: bool = False,
):
    """Evaluate a listener model p(reward | utt, options) on FlightPref games.

    Args:
        model: listener model
        games: games to evaluate the model on
        held_out_option_sets: array of unseen option sets to evaluate model posterior
        save_dir: directory to save results to
    """
    results = []
    eval_results = {
        metric: {num_utts: [] for num_utts in range(1, 7)}
        for metric in ["acc", "reward_l2"]
    }
    total_rnd_idx = -1

    for game_idx, g in enumerate(games):
        model.reset()
        true_reward = g["reward_function"]
        num_utts_observed = 0
        for rnd_idx, rnd in enumerate(g["rounds"]):
            # Only evaluate on good utterances that are informative enough for L to select correct option
            if rnd["utterance"] is None or not rnd["final_correct"]:
                continue
            total_rnd_idx += 1
            num_utts_observed += 1

            utt, opts = rnd["utterance"], rnd["options"]

            if isinstance(model, OracleListener):
                model.observe_reward(true_reward)
            else:
                model.observe(utt, opts)
            posterior_samples = model.sample_posterior()

            if debug:
                print(total_rnd_idx)
                print(f"{game_idx}, {rnd_idx}, {num_utts_observed}")
                print(utt)
                print(opts)
                print(posterior_samples.mean(axis=0))

            rnd_results = {
                "num_utts_observed": num_utts_observed,
                "posterior_mean": posterior_samples.mean(axis=0),
                "posterior_std": posterior_samples.std(axis=0),
                "game_idx": game_idx,
                "game_round_idx": rnd_idx,
            }
            results.append(rnd_results)

            avg_regret, avg_acc = sim_regret(
                rnd_results["posterior_mean"],
                true_reward,
                eval_sets=held_out_option_sets,
            )
            reward_l2 = np.linalg.norm(
                rnd_results["posterior_mean"] - true_reward, ord=2
            )
            eval_results["acc"][num_utts_observed].append(avg_acc)
            eval_results["reward_l2"][num_utts_observed].append(reward_l2)

            # Log.
            wandb.log(
                {
                    f"eval/avg_acc_{num_utts_observed}": avg_acc,
                    f"eval/running_avg_acc_{num_utts_observed}": np.mean(
                        eval_results["acc"][num_utts_observed]
                    ),
                }
            )

    print("Done.")
    with open(f"{save_dir}/results_games_REFACTOR.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{save_dir}/eval_results_games.pkl", "wb") as f:
        pickle.dump(eval_results, f)

    return results, eval_results


def load_listeners(num_models):
    print(f"LOADING {num_models} listener models")

    ds_args = dict(
        add_feature_extremes=True,
        unique_extremes=False,
        for_listener=True,
    )

    listener_models = []
    for seed in range(1, num_models + 1):
        _listener = BertEncoderOptionListener(
            feature_extremes=ds_args["add_feature_extremes"],
            unique_extremes=ds_args["unique_extremes"],
            choose_option_with_dot_product=True,
        )

        model_name = f"{REPO_PATH}/ckpts/bert-listener_hs-768_lr-2e-5_es-256_{seed}"
        print(f"loading {model_name}")
        load_model(_listener, model_name)
        _listener.cuda()
        _listener.eval()
        listener_models.append(_listener)
    return listener_models


def load_speakers(num_models):
    print(f"LOADING {num_models} speaker models")

    speaker_models = []
    for seed in range(1, num_models + 1):
        _speaker = LiteralEmbeddingSpeaker(
            hidden_size=512,
            feat_embed_size=128,
            reward_embed_size=128,
            per_feat_max_reward=True,
            feature_extremes=True,
            only_first_option=True,
            latent_type="fuse_scores",
            latent_reward_weight_priors="uniform_constant",
            language_rep="bert-base",
            no_mdp_attention=True,
            dot_product_scorer=True,
        )
        model_name = f"{REPO_PATH}/ckpts/embedding_s0_bert-base_no-mdp-attention_dps_lr-5e-5_latent-learned_shuf-neg-4_tbs-32_{seed}"

        load_model(_speaker, model_name)
        _speaker.cuda()
        _speaker.eval()
        speaker_models.append(_speaker)
    return speaker_models


if __name__ == "__main__":
    import sys

    print(" ".join(sys.argv))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Disable tqdm
    from functools import partialmethod

    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--experiment_name", default="test", help="basename to save results in results/"
    )
    parser.add_argument("--num_ensemble_models", type=int, default=5)
    parser.add_argument(
        "--model_type",
        choices=["oracle", "s0-reward-listener"],
        default="s0-reward-listener",
    )
    # General s1 settings
    parser.add_argument("--speaker_beta", type=float, default=1.0)
    # L2 settings
    parser.add_argument(
        "--s1_variant",
        choices=["log-interpolate", "interpolate"],
        default="interpolate",
    )
    parser.add_argument("--nearsightedness_lambda", type=float)
    parser.add_argument(
        "--normalize_per_term", dest="normalize_per_term", action="store_true"
    )

    parser.add_argument(
        "--no_normalize_per_term", dest="normalize_per_term", action="store_false"
    )
    # Oracle settings
    parser.add_argument("--oracle_num_uniform_feats", type=int)
    # S0 reward listener settings
    parser.add_argument("--temperature", type=float)

    args = parser.parse_args()
    print("**arguments:")
    pprint.pprint(vars(args))

    start_time = time.time()

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    pyro.set_rng_seed(args.seed)

    # Set up models
    if args.model_type != "oracle":
        listener_models = load_listeners(args.num_ensemble_models)
    print(args.num_ensemble_models)

    if args.model_type == "oracle":
        model = OracleListener(num_uniform_feats=args.oracle_num_uniform_feats)
    elif args.model_type == "s0-reward-listener":
        speaker_models = load_speakers(args.num_ensemble_models)
        model = L0AndS0PragmaticListener(
            listener_models,
            speaker_models,
            s1_variant=args.s1_variant,
            nearsightedness_lambda=args.nearsightedness_lambda,
            normalize_per_term=args.normalize_per_term,
            farsighted_temperature=args.temperature,
            cuda=True,
        )

    print("MODEL: ")
    print(model.__dict__)
    print()

    dump_git_status()

    # Evaluate on the same eval sets for all models.
    if not os.path.isfile("evaluate/eval_sets.npy"):
        # Generate eval sets
        eval_sets = [generate_options_arr() for _ in range(1000)]
        np.save("evaluate/eval_sets.npy", eval_sets)
    eval_sets = np.load("evaluate/eval_sets.npy")

    # Load evaluation games.
    with open("data/eval.json", "r") as f:
        val_games = json.load(f)
    print("Total num games: ", len(val_games))

    save_path = f"results/{args.experiment_name}/"
    os.makedirs(save_path, exist_ok=True)

    wandb.init(
        project="inferring-rewards",
        group="posterior-evaluation",
        config=vars(args),
        save_code=True,
        name=args.experiment_name,
    )

    run_model_on_games(model, val_games, eval_sets, save_path)
    print("TOTAL TIME: ", timedelta(seconds=time.time() - start_time))
