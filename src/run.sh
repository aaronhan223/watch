#!/bin/bash

python -W ignore main.py \
    --dataset0 wave \
    --muh_fun_name RF \
    --d0_shift_type noise \
    --bias 0.0 \
    --plot_errors True \
    --n_seeds 1 \
    --errs_window 1 \
    --cs_type signed \
    --weights_to_compute none \
    --label_shift 0.8 \
    --noise_mu 0.2 \
    --noise_sigma 0.05