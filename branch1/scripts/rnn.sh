#!/bin/bash

python3 -m branch1.main.rnn \
    --train_path "branch1/data_enhanced/training/train_lowfrict.csv" \
    --val_path "branch1/data_enhanced/training/val_lowfrict.csv" \
    --test_path "branch1/data_enhanced/training/test_lowfrict.csv" \
    --model_path "branch1/models/integrator/" \
    --model_name "integrator_model.pth" \
    --version "v1" \
    --epochs 500 \
    --patience 15 \
    --lr 0.01 \
    --wd 0.0 \
    --batch_size 512 \
    --optimizer "adam" \
    --l1_ref 1.0 \
    --l2_ref 5.0 \
    --m1_ref 1.0 \
    --m2_ref 1.0 \
    --b1_ref 0.1 \
    --b2_ref 0.1 \
    --training \
    --verbose \
    --show_plots
