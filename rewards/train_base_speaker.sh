#!/bin/bash

embed_size=128
hidden_size=512
dropout_p=0.2
train_batch_size=32
num_hard_negatives=4
bert_lr=5e-5
latent_priors=learned

for trial in `seq 1 5`
do
    ./run.sh embedding_s0_bert-base_no-mdp-attention_dps_lr-${bert_lr}_latent-${latent_priors}_shuf-neg-${num_hard_negatives}_tbs-${train_batch_size}_${trial} \
        --feat_embed_size $embed_size \
        --reward_embed_size $embed_size \
        --features per_feature_max_reward feature_extremes \
        --language_rep bert-base \
        --bert_learning_rate $bert_lr \
        --num_hard_negatives $num_hard_negatives \
        --train_batch_size $train_batch_size \
        --only_first_option \
        --hidden_size $hidden_size \
        --dropout_p $dropout_p \
        --latent_type=fuse_scores \
        --latent_reward_weight_priors=${latent_priors} \
        --embedding_speaker_no_mdp_attention \
        --embedding_speaker_dot_product_scorer
done
