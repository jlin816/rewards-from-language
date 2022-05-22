# Inferring Rewards From Language in Context

This repository contains code and datasets for the paper:

**Inferring Rewards from Language in Context**<br/>
Jessy Lin, Daniel Fried, Dan Klein, and Anca Dragan<br/>
ACL 2022<br/>
[[Paper]](https://arxiv.org/abs/2204.02515)<br/>

![Illustration of FlightPref dataset](/assets/flightpref.png)

**Abstract**: In classic instruction following, language like "I'd like the JetBlue flight" maps to actions (e.g., selecting that flight). However, language also conveys information about a user's underlying reward function (e.g., a general preference for JetBlue), which can allow a model to carry out desirable actions in new contexts. We present a model that infers rewards from language pragmatically: reasoning about how speakers choose utterances not only to elicit desired actions, but also to reveal information about their preferences. On a new interactive flight-booking task with natural language, our model more accurately infers rewards and predicts optimal actions in unseen environments, in comparison to past work that first maps language to actions (instruction following) and then maps actions to rewards (inverse reinforcement learning).

## Installation

1. conda is recommended for installation: `conda create -n rewards python=3.8`.
2. Clone this repository via: `git clone https://github.com/jlin816/rewards-from-language`
3. Run `pip install -e .` from the project root. 
4. Set `REPO_PATH` in `constants.py`.

## FlightPref Dataset

`data/` contains the FlightPref dataset, split into training and evaluation sets.

`train.json` and `eval.json` are grouped into games corresponding to the original data collection procedure, where each game has the same reward function. Each game is a JSON object with the following fields:
```
{
    "reward_function": [0.5, 0.5, 0.5, -0.5, 0, 0.5, 0, 0],
    "rounds": [
        <list of rounds>
    ]
}
``` 

`train.jsonl` is the same as `train.json`, but flattened into rounds for training, where each line is a JSON object with the following fields:
```
{
    "utterance": "one stop that is short", 
    "options": [[0.04, 0.0, 1.0, 0.0, 0.0, 0.52, 0.25, 0.0], [0.24, 0.0, 0.0, 1.0, 0.0, 0.4, 0.5, 0.35], [0.72, 0.0, 0.0, 0.0, 1.0, 0.64, 0.25, 0.48]],
    "optimal_index": 0,
    "reward_weights": [0.5, 0.5, 0.5, -0.5, 0, 0.5, 0, 0]
}
```

The features of the options and reward weights are (also defined in `constants.py`): 
```
[
    "arrival time before meeting"
    "american"
    "delta"
    "jetblue"
    "southwest"
    "longest stop"
    "number of stops"
    "price"
]
```

For more details on the dataset and data collection procedure, please refer to the paper.

## Usage

To train the base listener and speaker models, run:
```bash
sh train_base_listener.sh
sh train_base_speaker.sh
```

Each model seed should train in < 30 minutes.

To use the trained base models for inference, we have defined pragmatic models in `posterior_models/` that use the base speaker probabilities to calculate distributions over rewards given utterances. 

`nearsightedness_lambda` corresponds to the alpha parameter in the paper.

To evaluate the pragmatic models defined in the paper on the evaluation set of FlightPref, run the evaluation script (after changing the model paths to the paths for your trained base models):

```bash
sh evaluate/scripts/eval_games_all.sh
```

## Citation

If you find this code or our paper useful for your research, please consider citing the following:
```
@inproceedings{lin-etal-2022-inferring,
    title = "Inferring Rewards from Language in Context",
    author = "Lin, Jessy  and
      Fried, Daniel  and
      Klein, Dan  and
      Dragan, Anca",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.585",
    pages = "8546--8560"
}
```
