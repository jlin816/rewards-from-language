#!/bin/bash
experiment_name="$1"
shift

mkdir -p logs

python -u trainer.py \
  --cuda \
  --experiment_name ${experiment_name} \
  --num_epochs 20 \
  $@ \
  2>&1 \
  | tee logs/${experiment_name}.out
