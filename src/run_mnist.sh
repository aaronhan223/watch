#!/bin/bash

for corruption in brightness canny_edges dotted_line fog glass_blur identity impulse_noise motion_blur rotate scale shear shot_noise spatter stripe translate zigzag
do
    python -W ignore main_mnist_cifar.py \
        --dataset0 mnist \
        --dataset1 mnist_c \
        --n_seeds 3 \
        --cs_type probability \
        --methods baseline \
        --schedule variable \
        --epochs 3 \
        --lr 0.001 \
        --bs 64 \
        --init_phase 500 \
        --alpha 0.1 \
        --corruption_type $corruption \
        --severity 1 \
        --mixture_ratio_val 0.1 \
        --mixture_ratio_test 0.1 \
        --val_set_size 100 \
        --errs_window 1 \
        --sr_threshold 1e8 \
        --mt_threshold 1e8 \
        --verbose 
done
