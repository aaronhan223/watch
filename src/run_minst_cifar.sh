#!/bin/bash

python -W ignore main_mnist_cifar.py \
    --dataset0 mnist \
    --dataset1 mnist_c \
    --n_seeds 1 \
    --cs_type probability \
    --methods baseline \
    --schedule variable \
    --epochs 2 \
    --lr 0.001 \
    --bs 64 \
    --init_phase 500 \
    --corruption_type fog \
    --severity 5 \
    --train_test_split_only \
    --errs_window 1
