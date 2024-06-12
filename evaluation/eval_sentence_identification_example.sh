#!/bin/sh

GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate_sentence_identification.py \
    --model_dir 'google-bert/bert-base-multilingual-cased' \
    --pt_dir './result/mbert-sentence-layerwise-cls' \ 
    --test_dir '../data/sentence_identification_test.json' \
    --output_dir '../output/sentence_identification_result' \
    --probe_type 'layer_wise' \ # 'layer_wise' or 'cumulative_scoring'
    --remove_identity 'no' \ # 'no' or 'center' or 'pcr'
    --padding  \
    --truncation \
    --shuffle 
