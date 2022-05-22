import logging

import jsonlines
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from transformers import BertTokenizer

from rewards.constants import NUM_REWARD_FEATS, BERT_PAD_ID 
from rewards.synthetic import flip
from rewards.utils import (
    augment_options_with_indicators,
    augment_options_with_indicators_listener,
    deaugment_options,
    get_carrier_from_list,
    optimal_index,
)

tok = BertTokenizer.from_pretrained("bert-base-uncased")

logger = logging.getLogger(__name__)


class FlightTaskDataset(data.Dataset):
    def __init__(
        self,
        path: str,
        add_per_feat_max_reward=True,
        add_feature_extremes=True,
        unique_extremes=True,
        min_utterance_length=0,
        for_listener=False,
    ):
        self.samples = []
        self.add_per_feat_max_reward = add_per_feat_max_reward
        self.add_feature_extremes = add_feature_extremes
        self.unique_extremes = unique_extremes
        self.for_listener = for_listener

        with jsonlines.open(path) as reader:
            for ix, sample in enumerate(reader):
                if (
                    min_utterance_length
                    and len(self.tokenizer(sample["utterance"])) < min_utterance_length
                ):
                    continue
                reward = np.array(sample["reward_weights"])
                options = np.array(sample["options"])
                if self.for_listener:
                    sample["options"] = augment_options_with_indicators_listener(
                        deaugment_options(options),
                        add_feature_extremes=add_feature_extremes,
                        unique_extremes=unique_extremes,
                        return_type="numpy",
                    )

                    # Permute options for listener since data always has opt_index=0 by default
                    sample["options"] = np.random.permutation(sample["options"])
                    sample["optimal_index"] = optimal_index(
                        sample["reward_weights"],
                        sample["options"][:, :NUM_REWARD_FEATS],
                    )
                else:
                    sample["options"] = augment_options_with_indicators(
                        deaugment_options(options),
                        reward,
                        add_per_feat_max_reward=add_per_feat_max_reward,
                        add_feature_extremes=add_feature_extremes,
                        unique_extremes=unique_extremes,
                    )
                sample["index"] = ix
                self.samples.append(sample)

        logging.info("VOCAB: using pretrained BERT vocab and tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _encode_utterances(self, utts, **pretrained_tokenizer_kwargs):
        # transformers tokenizers usually add special tokens by default
        return self.tokenizer(utts, return_tensors="pt", **pretrained_tokenizer_kwargs)[
            "input_ids"
        ]

    def _decode_single_utterance(self, utt):
        return self.tokenizer.decode(utt, skip_special_tokens=True)

    def _decode_utterances(self, utts):
        return [self._decode_single_utterance(utt) for utt in utts]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Returns a single sample.
        Args:
            index (int or slice)

        Returns:
            utterance ((list of) list of int ids): encoded text
            reward_weights ((list of) list of ints)
            options ((list of) list of options, each a list of ints)
        """
        if isinstance(index, slice):
            samples = self.samples[index]
            utterances = self._encode_utterances(
                [sample["utterance"] for sample in samples]
            )
            reward_weights = [s["reward_weights"] for s in samples]
            options = [s["options"] for s in samples]
        elif isinstance(index, int):
            sample = self.samples[index]
            utterances = self._encode_utterances([sample["utterance"]])[0]
            reward_weights = sample["reward_weights"]
            options = sample["options"]

        if self.for_listener:
            return utterances, reward_weights, options, sample["optimal_index"]
        else:
            return utterances, reward_weights, options


class FlightTaskDatasetWithNegatives(FlightTaskDataset):
    def __init__(self, *args, num_hard_negatives=7, **kwargs):
        super(FlightTaskDatasetWithNegatives, self).__init__(*args, **kwargs)
        self.num_hard_negatives = num_hard_negatives

    def _get_hard_negatives(self, optimal_option, reward_wts, utt):
        """Returns a function that assembles synthetic hard negatives for the embedding
        speaker as encoded tensors by describing the negative of the reward function.
        """
        hard_negatives = []

        carrier = str.lower(get_carrier_from_list(optimal_option)[0])
        # hard_negatives.append(f"i don't like {carrier}")
        hard_negatives.append(f"")

        decoded_utt = self._decode_single_utterance(utt)
        hard_negatives.extend(flip(decoded_utt))

        if len(hard_negatives) > self.num_hard_negatives:
            hard_negatives = hard_negatives[: self.num_hard_negatives]
        elif len(hard_negatives) < self.num_hard_negatives:
            hard_negatives = hard_negatives + [hard_negatives[-1]] * (
                self.num_hard_negatives - len(hard_negatives)
            )

        return self._encode_utterances(hard_negatives, padding=True)

    def __getitem__(self, index):
        utt, rw, options = super().__getitem__(index)
        hard_negatives = self._get_hard_negatives(options[0], rw, utt)
        return utt, rw, options, hard_negatives


def collate_rewards_and_options(data):
    rw_batch, options_batch = [], []
    for rw, options in data:
        rw_batch.append(rw)
        options_batch.append(options)
    return {
        "reward_weights": torch.tensor(rw_batch, dtype=torch.float),
        "options": torch.tensor(options_batch, dtype=torch.float),
    }


def collate_fn_bert(data, cuda=False):
    data = list(data)
    collated_utts = pad_sequence(
        [utt for utt, _, _, _ in data], batch_first=True, padding_value=BERT_PAD_ID
    )
    collated_rew_opt = collate_rewards_and_options(
        (rw, opts) for _, rw, opts, _ in data
    )
    d = dict(**collated_rew_opt)
    d["utterance"] = collated_utts
    d["optimal_index"] = torch.tensor([x[3] for x in data], dtype=torch.long)
    if cuda:
        d = {k: v.cuda() for k, v in d.items()}
    return d


def collate_fn_embedding_speaker(data, is_bert=True):
    utt_batch, rw_batch, options_batch, hard_negs_batch_flat = [], [], [], []
    num_hard_negs = len(data[0][-1])
    utt_lengths = []
    hard_neg_lengths = []
    for (utt, rw, options, hard_negatives) in data:
        utt_batch.append(torch.tensor(utt, dtype=torch.long))
        utt_lengths.append(len(utt))
        rw_batch.append(rw)
        options_batch.append(options)
        assert (
            len(hard_negatives) == num_hard_negs
        ), "Number of hard negatives must be the same for each batch elem"
        hard_negs_batch_flat.extend(
            [torch.tensor(neg, dtype=torch.long) for neg in hard_negatives]
        )
        hard_neg_lengths.append([len(neg) for neg in hard_negatives])
    # Pad everything to same max length so we can feed all batch utts and hard negatives in at once
    # utt_batch: (batch,)
    # hard_negs_batch_flat: (batch * num_hard_negs)
    # (max_seq_len, batch * 2)
    batch_size = len(data)
    utt_and_negs_batch = pad_sequence(
        [*utt_batch, *hard_negs_batch_flat],
        batch_first=is_bert,
        padding_value=BERT_PAD_ID,
    )
    if is_bert:
        utt_batch = utt_and_negs_batch[:batch_size]
        _, max_seq_len = utt_batch.shape
        hard_negs_batch = (utt_and_negs_batch[batch_size:]).reshape(
            batch_size, num_hard_negs, max_seq_len
        )
    else:
        utt_batch = utt_and_negs_batch[:, :batch_size]
        max_seq_len, _ = utt_batch.shape
        hard_negs_batch = (utt_and_negs_batch[:, batch_size:]).reshape(
            max_seq_len, batch_size, num_hard_negs
        )
    return {
        "utterance": utt_batch,
        "utterance_length": torch.tensor(utt_lengths, dtype=torch.long),
        "reward_weights": torch.tensor(rw_batch, dtype=torch.float),
        "options": torch.tensor(options_batch, dtype=torch.float),
        "hard_negatives": hard_negs_batch,
        "hard_negatives_lengths": torch.tensor(hard_neg_lengths, dtype=torch.long),
    }


def create_dataloaders(
    dataset,
    train_prop,
    train_batch_size=64,
    val_batch_size=64,
    val_split=0.2,
    shuffle_train=True,
    collate_fn=collate_fn_bert,
    train_subset_fraction=None,
    show_info=True,
):
    """Creates train/val dataloaders."""
    num_val = int(len(dataset) * val_split)
    num_train = len(dataset) - num_val
    val_idxs = list(range(num_train, len(dataset)))
    num_train = int(num_train * train_prop)
    train_idxs = list(range(num_train))

    train_ds = data.Subset(dataset, train_idxs)
    val_ds = data.Subset(dataset, val_idxs)

    if show_info:
        logging.info(f"Train: {len(train_ds)} examples / Val: {len(val_ds)} examples")

    train_dl = data.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
    )
    val_dl = data.DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn
    )

    d = {"train": train_dl, "val": val_dl}

    if train_subset_fraction:
        train_subset_idxs = list(range(0, num_train, int(1 / train_subset_fraction)))
        train_subset_ds = data.Subset(dataset, train_subset_idxs)
        logging.info(f"Train Subset: {len(train_subset_ds)}")
        d["train_subset"] = data.DataLoader(
            train_subset_ds,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    return d
