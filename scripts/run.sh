#!/bin/bash

python -W ignore main.py \
    --dataset0 white_wine \
    --muh_fun_name RF \
    --d0_shift_type covariate \
    --bias 0.53 \
    --plot_errors True \
    --depth 3 \
    --n_seeds 1 \
    --errs_window 1 \
    --cs_type signed \
    --methods fixed_cal baseline \
    --label_shift 0.8 \
    --noise_mu 0.2 \
    --noise_sigma 0.05