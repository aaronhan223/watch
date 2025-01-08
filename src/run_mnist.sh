#!/bin/bash

for corruption in brightness canny_edges dotted_line fog glass_blur identity impulse_noise motion_blur rotate scale shear shot_noise spatter stripe translate zigzag
do
    python -W ignore main_mnist_cifar.py \
        --dataset0 mnist \
        --dataset1 mnist_c \
        --n_seeds 1 \
        --cs_type neg_log \
        --methods baseline \
        --schedule variable \
        --epochs 5 \
        --lr 0.001 \
        --bs 64 \
        --init_phase 500 \
        --corruption_type $corruption \
        --severity 1 \
        --train_test_split_only \
        --errs_window 1
done
