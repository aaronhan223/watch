#!/bin/bash

for val_ratio in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
do
    for test_ratio in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
    do
        python -W ignore ../src/main_mnist_cifar.py \
            --dataset0 mnist \
            --dataset1 mnist_c \
            --n_seeds 1 \
            --cs_type probability \
            --methods fixed_cal_offline none \
            --schedule variable \
            --epochs 30 \
            --weight_epoch 30 \
            --lr 0.001 \
            --bs 64 \
            --init_clean 100 \
            --init_corrupt 100 \
            --alpha 0.1 \
            --corruption_type brightness \
            --severity 1 \
            --mixture_ratio_val $val_ratio \
            --mixture_ratio_test $test_ratio \
            --val_set_size 100 \
            --errs_window 50 \
            --num_samples_vis 600 \
            --plot_image_data mnist_600_
    done
done