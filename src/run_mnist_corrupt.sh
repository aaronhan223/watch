#!/bin/bash

for corruption in brightness canny_edges dotted_line fog glass_blur identity impulse_noise motion_blur rotate scale shear shot_noise spatter stripe translate zigzag
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
        --init_clean 1 \
        --init_corrupt 1 \
        --alpha 0.1 \
        --corruption_type $corruption \
        --severity 1 \
        --mixture_ratio_val 0.1 \
        --mixture_ratio_test 0.4 \
        --val_set_size 100 \
        --errs_window 1 \
        --num_samples_vis 600 
done
