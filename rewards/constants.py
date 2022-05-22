from enum import Enum

REPO_PATH = "/path/to/rewards-from-language/rewards"  # CHANGE ME

# FlightPref task constants
CARRIERS = ["American", "Delta", "JetBlue", "Southwest"]
REWARD_KEYS = [
    "arrival_time_before_meeting",
    "carrier=American",
    "carrier=Delta",
    "carrier=JetBlue",
    "carrier=Southwest",
    "longest_stop",
    "num_stops",
    "price",
]
REWARD_KEYS_READABLE = [
    "arrival time before meeting",
    "american",
    "delta",
    "jetblue",
    "southwest",
    "longest stop",
    "number of stops",
    "price",
]

REWARD_KEYS_TO_CARRIER = {
    "carrier=American": "American",
    "carrier=Delta": "Delta",
    "carrier=JetBlue": "JetBlue",
    "carrier=Southwest": "Southwest",
}

REWARD_VALS = [-1, -0.5, 0, 0.5, 1]
NUM_REWARD_FEATS = 8
# Feats:
#   `num_reward_feats` feats for each feat value
#   `num_reward_feats` indicators, 1 if option has max reward for that value
#   1 indicator for whether it is the optimal option
NUM_FEATS = NUM_REWARD_FEATS * 2 + 1


class FeatureType(Enum):
    CATEGORICAL = 0
    NUMERICAL = 1


OPTION_FEATURES = {
    "price": {"type": FeatureType.NUMERICAL, "values": [0, 100]},
    "carrier": {
        "type": FeatureType.CATEGORICAL,
        "values": ["American", "Delta", "JetBlue", "Southwest"],
    },
    "num_stops": {"type": FeatureType.NUMERICAL, "values": [0, 4]},
    "longest_stop": {"type": FeatureType.NUMERICAL, "values": [0, 25]},
    "arrival_time_before_meeting": {"type": FeatureType.NUMERICAL, "values": [0, 25]},
}
NUM_OPTIONS = 3

# Model and training constants
EPS = 1e-3
PAD = "<pad>"
BERT_PAD_ID = 0
BERT_BASE_HS = 768
