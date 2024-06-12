#!/bin/sh

GPU_ID=0
# when run roberta-large model set range as {1..24}
for i in {1..12}
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 sentence_identification.py \
    	--model_dir 'google-bert/bert-base-multilingual-cased' \
    	--train_dir '../data/sentence_identification_train.json' \
        --valid_dir '../data/sentence_identification_valid.json' \
    	--output_dir '../result/mbert-sentence-layerwise-cls' \
        --probe_type 'layer_wise' \  # 'layer_wise' or 'cumulative_scoring'
    	--epochs 1 \
        --probe_layer $i \
        --pooler 'cls' \ # 'cls' or 'avg'
        --padding  \
        --truncation \
        --shuffle 
done
