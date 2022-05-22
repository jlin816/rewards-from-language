from collections import defaultdict

import torch
import torch.nn.functional as F

import pprint
import functools
from typing import Optional
from torch.utils.data import DataLoader
import logging
import wandb
import numpy as np

import rewards.utils as utils
from rewards.constants import NUM_REWARD_FEATS, REPO_PATH
from rewards.utils import save_model, save_config

logger = logging.getLogger(__name__)


def train_embedding_speaker(
    model,
    train_dl: DataLoader,
    val_dl: DataLoader,
    save_name: str,
    num_epochs: int = 100,
    cuda: bool = False,
    train_subset_dl: Optional[DataLoader] = None,
    topk: int = 10,
    is_bert: bool = False,
    bert_learning_rate: float = 2e-5,
    infonce: bool = False,
    infonce_latent_posterior_weighting: bool = False,
    infonce_reward_only: bool = False,
):
    """
    Args:
        model;
        train_dl, val_dl: iterable torch dataloaders for train / val set
    """
    if is_bert:
        optimizer = torch.optim.AdamW(model.parameters(), lr=bert_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
    else:
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = None

    if cuda:
        model.cuda()
    model.train()
    best_val_topk_acc = 0
    best_log_info = None
    save_config(model, save_name)

    for i in range(num_epochs):
        train_loss = 0
        train_stats = defaultdict(list)

        for batch_ix, batch in enumerate(train_dl):

            assert model.training
            optimizer.zero_grad()

            # Load batch data.
            if cuda:
                batch = {k: v.cuda() for k, v in batch.items()}
            text, reward_weights, options = (
                batch["utterance"],
                batch["reward_weights"],
                batch["options"],
            )
            hard_negatives = batch["hard_negatives"]

            # Take grad step.
            loss = model.compute_loss(
                options,
                reward_weights,
                text,
                hard_negatives,
                infonce=infonce,
                infonce_latent_posterior_weighting=infonce_latent_posterior_weighting,
                infonce_reward_only=infonce_reward_only,
            )
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update and log train stats.
            this_stats = dict(
                loss=loss.item(),
                non_bert_param_norm=sum(
                    param.norm().item() for param in model.parameters()
                ),
                non_bert_grad_norm=sum(
                    param.grad.norm().item() if param.grad is not None else 0
                    for param in model.parameters()
                ),
            )
            if hasattr(model, "bert_encoder"):
                this_stats.update(
                    dict(
                        bert_param_norm=sum(
                            param.norm().item()
                            for param in model.bert_encoder.parameters()
                        ),
                        bert_grad_norm=sum(
                            param.grad.norm().item() if param.grad is not None else 0
                            for param in model.bert_encoder.parameters()
                        ),
                    )
                )
                this_stats["non_bert_param_norm"] -= this_stats["bert_param_norm"]
                this_stats["non_bert_grad_norm"] -= this_stats["bert_grad_norm"]

            for k, v in this_stats.items():
                train_stats[k].append(v)

            wandb.log({f"train/{k}": v for k, v in train_stats.items()})

            # Print and reset train stats.
            if (batch_ix - 1) % 20 == 0:
                agg_stats = {k: np.mean(v) for k, v in train_stats.items()}
                print("\t".join(f"{k}:{v:.3f}" for k, v in sorted(agg_stats.items())))
                train_stats = defaultdict(list)

        # Validate at end of epoch.
        validate_as_listener = infonce and infonce_reward_only
        if validate_as_listener:
            acc_str = "listener acc"
        else:
            acc_str = "acc"
        val_loss, val_acc, val_topk_acc = validate_embedding_speaker(
            model, val_dl, cuda=cuda, validate_as_listener=validate_as_listener
        )
        train_loss = train_loss / len(train_dl)
        log_info = [
            f"Epoch {i}: lr {optimizer.param_groups[0]['lr']} / train {train_loss:.4f} /"
        ]

        # Additionally evaluate on a subset of train.
        if train_subset_dl is not None:
            (
                train_subset_loss,
                train_subset_acc,
                train_subset_topk_acc,
            ) = validate_embedding_speaker(
                model,
                train_subset_dl,
                cuda=cuda,
                topk=topk,
                validate_as_listener=validate_as_listener,
            )
            log_info.append(
                f"train_subset\tloss: {train_subset_loss: .4f}\t{acc_str}: {train_subset_acc:.4f}\ttopk-{topk} {acc_str}: {train_subset_topk_acc:.4f}"
            )
        log_info.append(
            f"val\tloss: {val_loss:.4f}\t{acc_str} {val_acc:.4f}\ttopk-{topk} {acc_str}: {val_topk_acc:.4f}"
        )

        wandb.log(
            {
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/topk_acc": val_topk_acc,
                "train/subset_loss": train_subset_loss,
                "train/subset_acc": train_subset_acc,
                "train/subset_topk_acc": train_subset_topk_acc,
            }
        )

        # Save best model based on topk accuracy.
        if val_topk_acc > best_val_topk_acc:
            print(
                f"Best val topk-{topk} {acc_str}, epoch {i}: {val_topk_acc:.4f}, saving to {save_name}"
            )
            best_val_topk_acc = val_topk_acc
            model.cpu()
            save_model(model, save_name)
            if cuda:
                model.cuda()
            best_log_info = "BEST: {}".format("\t".join(log_info))

        if scheduler is not None:
            scheduler.step(val_topk_acc)
        logging.info("=" * 20)
        logging.info(" // ".join(log_info))

    logging.info(f"Best val topk-{topk} acc: {best_val_topk_acc}")
    logging.info(best_log_info)
    print(best_log_info)
    return best_val_topk_acc


def validate_embedding_speaker(
    model, val_dl, cuda=False, topk=10, validate_as_listener=False
):
    model.eval()
    total_preds = 0
    correct_preds = 0
    correct_preds_topk = 0

    total_loss = 0

    if model.latent_type is not None:
        ws = model.latent_reward_weights
        ps = model.latent_reward_prior_logits.softmax(-1)
        prob_str = ",".join(
            f"{w:.2f}: {p:.4f}" for w, p in zip(ws.tolist(), ps.tolist())
        )
        print(f"reward weight probabilities: {prob_str}")

    with torch.no_grad():
        for batch in val_dl:
            if cuda:
                batch = {k: v.cuda() for k, v in batch.items()}
            text, reward_weights, options = (
                batch["utterance"],
                batch["reward_weights"],
                batch["options"],
            )
            hard_negatives = batch["hard_negatives"]
            batch_size = reward_weights.size(0)
            scores, extras = model.add_negatives_and_score_batch(
                options, reward_weights, text, hard_negatives
            )
            if validate_as_listener:
                log_probs = model.get_reward_only_logits(scores).log_softmax(0)
                argmax_dim = 0
            else:
                log_probs = model.marginalize_latents(
                    scores,
                )
                argmax_dim = -1
            pred_indices = log_probs.argmax(argmax_dim)
            # (batch, k)
            if log_probs.size(argmax_dim) < topk:
                print(f"warning: k {topk} > support size {log_probs.size(argmax_dim)}")
                k_to_take = log_probs.size(argmax_dim)
            else:
                k_to_take = topk
            pred_indices_topk = torch.topk(log_probs, k_to_take, argmax_dim).indices
            if argmax_dim == 0:
                assert pred_indices_topk.dim() == 2
                pred_indices_topk = pred_indices_topk[:, :batch_size]
                pred_indices = pred_indices[:batch_size]
            total_preds += batch_size
            correct_preds += (
                (pred_indices.detach().cpu() == torch.arange(batch_size)).sum().item()
            )
            correct_preds_topk += (
                torch.any(
                    pred_indices_topk.detach().cpu()
                    == torch.arange(batch_size).unsqueeze(argmax_dim),
                    dim=argmax_dim,
                )
                .sum()
                .item()
            )
            total_loss -= log_probs.diagonal(0).sum().item()

    accuracy = 100 * correct_preds / total_preds
    topk_acc = 100 * correct_preds_topk / total_preds
    model.train()
    return total_loss / len(val_dl), accuracy, topk_acc


if __name__ == "__main__":
    from data import (
        FlightTaskDataset,
        create_dataloaders,
        collate_fn_embedding_speaker,
        FlightTaskDatasetWithNegatives,
    )
    from models import LiteralEmbeddingSpeaker
    import sys

    print(" ".join(sys.argv))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        default="test",
        help="basename to save model file in ckpts/",
    )
    parser.add_argument(
        "--train_prop",
        type=float,
        default=1.0,
        help="fraction of the training data to use",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        choices=["per_feature_max_reward", "feature_extremes"],
        default=["per_feature_max_reward"],
        help="indicator features to add to option features. per_feature_max_reward: 1 iff that "
        "option's feature has the max reward across all options. feature_extremes: adds a "
        "min and max for each feature; 1 iff that feature value is the min/max",
    )
    parser.add_argument(
        "--unique_extremes",
        action="store_true",
        help="min/max indicator features are 1 iff the feature is the unique min or max",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--feat_embed_size", type=int, default=16)
    parser.add_argument("--reward_embed_size", type=int, default=8)
    parser.add_argument(
        "--language_pooling", choices=["max", "bidi", "self_attention"], default="max"
    )
    parser.add_argument("--dropout_p", type=float, default=0.4)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_hard_negatives", type=int, default=7)
    parser.add_argument("--only_first_option", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)

    # latent training
    parser.add_argument("--latent_type", choices=["fuse_scores"])
    parser.add_argument(
        "--latent_reward_weight_priors",
        choices=["uniform_constant", "learned"],
        default="uniform_constant",
    )
    parser.add_argument(
        "--latent_reward_weights", type=float, nargs="+", default=[0.0, 1.0]
    )

    # bert
    parser.add_argument("--language_rep", choices=["bert-base"], default="bert-base")
    parser.add_argument("--bert_learning_rate", type=float, default=2e-5)

    parser.add_argument("--embedding_speaker_no_mdp_attention", action="store_true")
    parser.add_argument("--embedding_speaker_dot_product_scorer", action="store_true")

    parser.add_argument("--embedding_speaker_infonce", action="store_true")
    parser.add_argument(
        "--embedding_speaker_infonce_latent_posterior_weighting", action="store_true"
    )
    parser.add_argument(
        "--embedding_speaker_infonce_reward_only",
        action="store_true",
        help="loss is only log p(r | u). makes most sense with --latent_type=fuse_scores and --latent_reward_weights=1.0",
    )

    args = parser.parse_args()
    print("**arguments:")
    pprint.pprint(vars(args))

    utils.dump_git_status()

    add_per_feat_max_reward = False
    add_feature_extremes = False
    for feature_type in args.features:
        if feature_type == "per_feature_max_reward":
            add_per_feat_max_reward = True
        elif feature_type == "feature_extremes":
            add_feature_extremes = True
        else:
            raise ValueError(f"invalid feature_type {feature_type}")

    unique_extremes = args.unique_extremes

    is_bert = False
    ds_args = dict(
        add_per_feat_max_reward=add_per_feat_max_reward,
        add_feature_extremes=add_feature_extremes,
        unique_extremes=unique_extremes,
    )
    ds_class = FlightTaskDatasetWithNegatives
    collate = collate_fn_embedding_speaker
    ds_args["num_hard_negatives"] = args.num_hard_negatives
    if args.language_rep == "bert-base":
        is_bert = True

    if is_bert:
        collate = functools.partial(collate_fn_embedding_speaker, is_bert=True)
    else:
        ds_args["vocab"] = FlightTaskDataset(f"{REPO_PATH}/data/train.jsonl").vocab

    train_ds = ds_class(f"{REPO_PATH}/data/train.jsonl", **ds_args)

    # use separate batch sizes for train and val because validation batch size affects the embedding speaker's top-k accuracy metric
    dataloaders = create_dataloaders(
        train_ds,
        args.train_prop,
        val_split=0.05,
        train_subset_fraction=0.2,
        collate_fn=collate,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
    )
    train_dl, val_dl, train_subset_dl = (
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["train_subset"],
    )

    num_features = NUM_REWARD_FEATS + 1  # +1 for global max
    if add_per_feat_max_reward:
        num_features += NUM_REWARD_FEATS
    if add_feature_extremes:
        num_features += NUM_REWARD_FEATS * 2

    wandb.init(
        project="inferring-rewards",
        group="train-embedding-speaker",
        config=vars(args),
        save_code=True,
        name=args.experiment_name,
    )

    speaker_model = LiteralEmbeddingSpeaker(
        hidden_size=args.hidden_size,
        dropout_p=args.dropout_p,
        vocab_size=None if is_bert else len(train_ds.vocab.vocab),
        feat_embed_size=args.feat_embed_size,
        reward_embed_size=args.reward_embed_size,
        language_pooling=args.language_pooling,
        per_feat_max_reward=add_per_feat_max_reward,
        feature_extremes=add_feature_extremes,
        unique_extremes=unique_extremes,
        only_first_option=args.only_first_option,
        latent_type=args.latent_type,
        latent_reward_weights=args.latent_reward_weights,
        latent_reward_weight_priors=args.latent_reward_weight_priors,
        language_rep=args.language_rep,
        dot_product_scorer=args.embedding_speaker_dot_product_scorer,
        no_mdp_attention=args.embedding_speaker_no_mdp_attention,
    )
    train_embedding_speaker(
        speaker_model,
        train_dl,
        val_dl,
        save_name=f"ckpts/{args.experiment_name}",
        num_epochs=args.num_epochs,
        cuda=args.cuda,
        train_subset_dl=train_subset_dl,
        topk=args.topk,
        is_bert=is_bert,
        bert_learning_rate=args.bert_learning_rate,
        infonce=args.embedding_speaker_infonce,
        infonce_latent_posterior_weighting=args.embedding_speaker_infonce_latent_posterior_weighting,
        infonce_reward_only=args.embedding_speaker_infonce_reward_only,
    )
