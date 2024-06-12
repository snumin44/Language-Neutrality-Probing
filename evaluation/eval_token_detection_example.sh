#!/bin/sh

GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate_token_detection.py \
    --model_dir 'google-bert/bert-base-multilingual-cased' \
    --pt_dir '../result/mbert-token-layerwise' \
    --test_dir '../data/token_detection_test.json' \
    --output_dir '../output/mbert-token-layerwise' \
    --probe_type 'layer_wise' \ # 'laywer_wise' or 'cumulative_scoring'
    --padding  \
    --truncation \
    --shuffle 