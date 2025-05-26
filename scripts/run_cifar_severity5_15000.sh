#!/bin/bash

for val_ratio in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
do
    for test_ratio in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
    do
        python -W ignore ../src/main_mnist_cifar.py \
            --dataset0 cifar10 \
            --dataset1 cifar10_c \
            --n_seeds 5 \
            --cs_type probability \
            --methods fixed_cal_offline none \
            --schedule variable \
            --epochs 30 \
            --weight_epoch 30 \
            --lr 0.001 \
            --bs 64 \
            --init_clean 5000 \
            --init_corrupt 5000 \
            --alpha 0.1 \
            --corruption_type brightness \
            --severity 5 \
            --mixture_ratio_val $val_ratio \
            --mixture_ratio_test $test_ratio \
            --val_set_size 5000 \
            --errs_window 200 \
            --num_samples_vis 15000 \
            --plot_image_data cifar_l5_15000_
    done
done 