#!/bin/bash

for corruption in brightness contrast defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur
do
    python -W ignore main_mnist_cifar.py \
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
        --corruption_type $corruption \
        --severity 5 \
        --mixture_ratio_val 0.1 \
        --mixture_ratio_test 0.3 \
        --val_set_size 100 \
        --errs_window 1 \
        --num_samples_vis 600 
done
