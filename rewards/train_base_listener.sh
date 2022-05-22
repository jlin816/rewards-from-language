#!/bin/bash

for seed in `seq 1 5`
do
  ./run_listener.sh bert-listener_hs-768_lr-2e-5_es-256_${seed} \
    --cuda \
    --hidden_size 768 \
    --learning_rate 2e-5 \
    --feat_embed_size 256 
done
