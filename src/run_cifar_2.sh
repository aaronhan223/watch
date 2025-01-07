#!/bin/bash

for corruption in brightness contrast defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur
do
    python -W ignore main_mnist_cifar.py \
        --dataset0 cifar10 \
        --dataset1 cifar10_c \
        --n_seeds 1 \
        --cs_type probability \
        --methods baseline \
        --schedule variable \
        --epochs 30 \
        --lr 0.001 \
        --bs 64 \
        --init_phase 500 \
        --corruption_type $corruption \
        --severity 2 \
        --train_test_split_only \
        --errs_window 1
done
