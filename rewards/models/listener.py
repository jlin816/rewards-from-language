import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

from rewards.constants import BERT_PAD_ID, NUM_OPTIONS, NUM_REWARD_FEATS
from rewards.models.encoders import OptionEncoder

BERT_BASE_HS = 768


class BertEncoderOptionListener(nn.Module):
    def __init__(
        self,
        feat_embed_size=256,
        hidden_size=768,
        dropout_p=0.1,
        encode_options_independently=True,
        choose_option_with_dot_product=True,
        feature_extremes=True,
        unique_extremes=False,
    ):
        """
        Args:
            encode_options_independently (bool):
                True: encode each option separately
                False: encode all options together
            choose_option_with_dot_product (bool):
                True: dot product encoded utt and encoded options to get choice
                False: concat repr of utt and all options and feed through MLP to get choice
        """
        super(BertEncoderOptionListener, self).__init__()
        self.feat_embed_size = feat_embed_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.encode_options_independently = encode_options_independently
        self.choose_option_with_dot_product = choose_option_with_dot_product
        self.utt_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.indicators = {
            "feature_extremes": feature_extremes,
            "unique_extremes": unique_extremes,
        }
        self.num_feats = NUM_REWARD_FEATS
        if self.indicators["feature_extremes"]:
            self.num_feats += NUM_REWARD_FEATS * 2

        logging.info(
            "Encoding options independently?: %s", self.encode_options_independently
        )
        logging.info(
            "Choosing with dot product (instead of flat MLP)?: %s",
            self.choose_option_with_dot_product,
        )

        if self.encode_options_independently:
            self.option_encoder = OptionEncoder(
                feat_embed_size=feat_embed_size,
                dropout_p=0.0,
                num_feats=self.num_feats,
            )
        else:
            # TODO: implement flat option encoder
            raise NotImplementedError

        if self.choose_option_with_dot_product:
            self.fc1 = nn.Linear(feat_embed_size, BERT_BASE_HS)
            self.fc1_dropout = nn.Dropout(self.dropout_p)
            self.fc2 = nn.Linear(BERT_BASE_HS, BERT_BASE_HS)
        else:
            self.fc1 = nn.Linear(
                feat_embed_size * NUM_OPTIONS + BERT_BASE_HS, hidden_size
            )
            self.fc1_dropout = nn.Dropout(self.dropout_p)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc2_dropout = nn.Dropout(self.dropout_p)
            self.fc3 = nn.Linear(hidden_size, NUM_OPTIONS)

    def encode_utt(self, utt):
        # (bs, bert_base_hs)
        attention_mask = utt != BERT_PAD_ID
        return self.utt_encoder(
            input_ids=utt, attention_mask=attention_mask
        ).pooler_output  # gets hs for first [CLS] token

    def forward(self, utt, options, encoded_utt=None):
        # (bs, bert_base_hs)
        if encoded_utt is None:
            encoded_utt = self.encode_utt(utt)
        # (bs, num_options, feat_embed_size)
        encoded_options = self.option_encoder(options)

        if self.choose_option_with_dot_product:
            # (bs, num_options, BERT_BASE_HS)
            x = self.fc1(encoded_options)
            x = self.fc1_dropout(x)
            x = F.relu(x)
            x = self.fc2(x)
            logits = torch.einsum("bij,bj->bi", (x, encoded_utt))
        else:
            encoded_options = encoded_options.view(
                encoded_options.shape[0], NUM_OPTIONS * self.feat_embed_size
            )
            x = torch.cat((encoded_utt, encoded_options), dim=-1)
            x = self.fc1(x)
            x = self.fc1_dropout(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.fc2_dropout(x)
            x = F.relu(x)
            logits = self.fc3(x)
        return logits
