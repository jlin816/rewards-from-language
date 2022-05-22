#!/bin/bash
experiment_name="$1"
shift

mkdir -p logs

python -u trainer_listener.py \
  --cuda \
  --experiment_name ${experiment_name} \
  $@ \
  2>&1 \
  | tee logs/${experiment_name}.out
