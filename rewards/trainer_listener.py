import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pprint
import logging
import tqdm
from typing import Optional
import wandb

from rewards.constants import REPO_PATH
from rewards.utils import save_model


def train_listener(
    model,
    train_dl: DataLoader,
    val_dl: DataLoader,
    save_name: str,
    num_epochs: int = 100,
    cuda: bool = False,
    train_subset_dl: Optional[DataLoader] = None,
    lr=2e-5,
):
    """
    Args:
        model;
        train_dl, val_dl: iterable torch dataloaders for train / val set
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if cuda:
        model.cuda()
    model.train()
    best_val_loss = 100

    for i in range(num_epochs):

        train_loss = 0
        stats = {}

        for batch in tqdm.tqdm(train_dl):

            assert model.training
            optimizer.zero_grad()

            # Load batch data.
            text, reward_weights, options, optimal_index = (
                batch["utterance"],
                batch["reward_weights"],
                batch["options"],
                batch["optimal_index"],
            )
            if cuda:
                text = text.cuda()
                reward_weights = reward_weights.cuda()
                options = options.cuda()
                optimal_index = optimal_index.cuda()

            # Take grad step.
            logits = model(text, options)
            loss = F.cross_entropy(logits, optimal_index)
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_dl)
        log_info = [f"Epoch {i}: / train {train_loss:.4f}"]
        stats["train/loss"] = train_loss

        # Additionally evaluate on train subset.
        if train_subset_dl is not None:
            train_subset_acc, train_subset_loss = validate_listener(
                model, train_subset_dl, cuda=cuda
            )
            stats.update(
                {
                    "train/subset_acc": train_subset_acc,
                    "train/subset_loss": train_subset_loss,
                }
            )
            log_info.append(
                f"train_subset loss {train_subset_loss: .4f} acc {train_subset_acc:.4f}"
            )

        val_acc, val_loss = validate_listener(model, val_dl, cuda=cuda)
        stats.update({"val/acc": val_acc, "val/loss": val_loss})
        log_info.append(f"val  loss {val_loss:.4f} acc {val_acc:.4f}")
        logging.info("=" * 20)
        logging.info(" // ".join(log_info))

        # Save best model.
        if val_loss < best_val_loss:
            print(f"Best val loss, epoch {i}: {val_loss}, saving to {save_name}")
            best_val_loss = val_loss
            save_model(model, save_name)

        wandb.log(stats)

        # Early stopping.
        if val_loss > 1.5 * best_val_loss:
            print("Early stopping.")
            break

    logging.info(f"Best val loss: {best_val_loss}")
    return best_val_loss


def validate_listener(model, val_dl, cuda=False):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_preds = 0

    with torch.no_grad():
        for batch in val_dl:
            text, reward_weights, options, optimal_index = (
                batch["utterance"],
                batch["reward_weights"],
                batch["options"],
                batch["optimal_index"],
            )
            total_preds += reward_weights.shape[0]
            if cuda:
                text = text.cuda()
                reward_weights = reward_weights.cuda()
                options = options.cuda()
                optimal_index = optimal_index.cuda()
            # (bsz, 3)
            option_logits = model(text, options)
            loss = F.cross_entropy(option_logits, optimal_index)
            total_loss += loss
            preds = option_logits.argmax(1)
            correct = (preds == optimal_index).type(torch.float32).sum().item()
            total_acc += correct
    loss = total_loss / len(val_dl)
    acc = 100 * total_acc / total_preds
    model.train()
    return acc, loss


if __name__ == "__main__":
    from data import FlightTaskDataset, create_dataloaders, collate_fn_bert
    from models import BertEncoderOptionListener

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
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--feat_embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()
    print("**arguments:")
    pprint.pprint(vars(args))

    ds_args = dict(add_feature_extremes=True, unique_extremes=False, for_listener=True)
    train_ds = FlightTaskDataset(f"{REPO_PATH}/data/train.jsonl", **ds_args)

    dataloaders = create_dataloaders(
        train_ds,
        1.0,
        val_split=0.05,
        train_subset_fraction=0.2,
        collate_fn=collate_fn_bert,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
    )
    train_dl, val_dl, train_subset_dl = (
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["train_subset"],
    )

    wandb.init(
        project="inferring-rewards",
        group="train-listener",
        config=vars(args),
        save_code=True,
        name=args.experiment_name,
    )

    listener = BertEncoderOptionListener(
        feat_embed_size=args.feat_embed_size,
        hidden_size=args.hidden_size,
        feature_extremes=ds_args["add_feature_extremes"],
        unique_extremes=ds_args["unique_extremes"],
        choose_option_with_dot_product=True,
    )
    train_listener(
        listener,
        train_dl,
        val_dl,
        save_name=f"ckpts/{args.experiment_name}",
        num_epochs=args.num_epochs,
        cuda=args.cuda,
        train_subset_dl=train_subset_dl,
        lr=args.learning_rate,
    )
