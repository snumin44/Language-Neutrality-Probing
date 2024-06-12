#!/bin/sh

GPU_ID=0
# when run roberta-large model set range as {1..24}
for i in {1..12}
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 token_detection.py \
    	--model_dir 'google-bert/bert-base-multilingual-cased' \
    	--train_dir '../data/token_detection_train.json' \
        --valid_dir '../data/token_detection_valid.json' \
    	--output_dir '../result/mbert-token-layerwise' \
        --probe_type 'layer_wise' \ # 'layer_wise' or 'cumulative_scoring'
    	--epochs 1 \
        --learning_rate 5e-3 \
        --probe_layer $i \
        --padding  \
        --truncation \
        --shuffle 
done