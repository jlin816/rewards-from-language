#!/bin/bash

exp_name="action_only"
mkdir -p results/${exp_name}
python -u evaluate/evaluate_l0.py \
  --experiment_name $exp_name \
  --model_type s0-reward-listener \
  --normalize_per_term \
  --s1_variant "interpolate" \
  --nearsightedness_lambda 1. \
  --num_ensemble_models 2 \
  --seed 0 \
  2>&1 \
  > results/${exp_name}/results_games.out

exp_name="reward_only"
mkdir -p results/${exp_name}
python -u evaluate/evaluate_l0.py \
  --experiment_name $exp_name \
  --model_type s0-reward-listener \
  --normalize_per_term \
  --s1_variant "interpolate" \
  --nearsightedness_lambda 0 \
  --num_ensemble_models 2 \
  --temperature 3 \
  --seed 0 \
  2>&1 \
  > results/${exp_name}/results_games.out

exp_name="ours"
mkdir -p results/${exp_name}
python -u evaluate/evaluate_l0.py \
  --experiment_name $exp_name \
  --model_type s0-reward-listener \
  --normalize_per_term \
  --s1_variant "interpolate" \
  --nearsightedness_lambda 0.5 \
  --temperature 3 \
  --num_ensemble_models 2 \
  --seed 0 \
  2>&1 \
  > results/${exp_name}/results_games.out
