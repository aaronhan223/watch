#!/bin/bash

# for ratio in 0.01 0.05 0.1 0.15 0.2 0.3 0.4
# do
#     python -W ignore main_mnist_cifar.py \
#         --dataset0 mnist \
#         --dataset1 mnist_c \
#         --n_seeds 1 \
#         --cs_type probability \
#         --methods fixed_cal_offline none \
#         --schedule variable \
#         --epochs 3 \
#         --lr 0.001 \
#         --bs 64 \
#         --init_phase 50 \
#         --alpha 0.1 \
#         --corruption_type brightness \
#         --severity 1 \
#         --mixture_ratio_val $ratio \
#         --mixture_ratio_test $ratio \
#         --val_set_size 400 \
#         --errs_window 1 \
#         --num_samples_vis 800 
# done

python -W ignore main_mnist_cifar.py \
    --dataset0 mnist \
    --dataset1 mnist_c \
    --n_seeds 1 \
    --cs_type probability \
    --methods fixed_cal_offline none \
    --schedule variable \
    --epochs 5 \
    --weight_epoch 80 \
    --lr 0.001 \
    --bs 64 \
    --init_phase 1 \
    --alpha 0.1 \
    --corruption_type brightness \
    --severity 1 \
    --mixture_ratio_val 0.1 \
    --mixture_ratio_test 0.4 \
    --val_set_size 100 \
    --errs_window 1 \
    --num_samples_vis 600 