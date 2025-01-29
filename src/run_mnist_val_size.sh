#!/bin/bash

for val_size in 50 100 200 400 600
do
    python -W ignore main_mnist_cifar.py \
        --dataset0 mnist \
        --dataset1 mnist_c \
        --n_seeds 10 \
        --cs_type probability \
        --methods fixed_cal_offline none \
        --schedule variable \
        --epochs 10 \
        --lr 0.001 \
        --bs 64 \
        --init_phase 50 \
        --alpha 0.1 \
        --corruption_type brightness \
        --severity 1 \
        --mixture_ratio_val 0.05 \
        --mixture_ratio_test 0.05 \
        --val_set_size $val_size \
        --errs_window 1 \
        --num_samples_vis 800 
done