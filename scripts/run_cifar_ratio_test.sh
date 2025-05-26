#!/bin/bash

for ratio in 0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5
do
    python -W ignore ../src/main_mnist_cifar.py \
        --dataset0 cifar10 \
        --dataset1 cifar10_c \
        --n_seeds 1 \
        --cs_type probability \
        --methods fixed_cal_offline none \
        --schedule variable \
        --epochs 30 \
        --weight_epoch 80 \
        --lr 0.001 \
        --bs 64 \
        --init_clean 50 \
        --init_corrupt 1 \
        --alpha 0.1 \
        --corruption_type brightness \
        --severity 1 \
        --mixture_ratio_val 0.1 \
        --mixture_ratio_test $ratio \
        --val_set_size 100 \
        --errs_window 1 \
        --num_samples_vis 600 \
        --plot_image_data cifar_l1_600_
