#!/bin/bash

for ip in 1 5 10 15 25 50
do
    python -W ignore main_mnist_cifar.py \
        --dataset0 mnist \
        --dataset1 mnist_c \
        --n_seeds 1 \
        --cs_type probability \
        --methods fixed_cal_offline none \
        --schedule variable \
        --epochs 10 \
        --weight_epoch 80 \
        --lr 0.001 \
        --bs 64 \
        --init_clean $ip \
        --init_corrupt 1 \
        --alpha 0.1 \
        --corruption_type brightness \
        --severity 1 \
        --mixture_ratio_val 0.1 \
        --mixture_ratio_test 0.4 \
        --val_set_size 100 \
        --errs_window 1 \
        --num_samples_vis 600 
done