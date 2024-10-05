#!/bin/bash

python main.py \
    --dataset0 white_wine \
    --muh_fun_name RF \
    --d0_shift_type noise \
    --bias 0.0 \
    --plot_errors True \
    --n_seeds 50 \
    --errs_window 50 \
    --cs_type signed \
    --label_shift 1 \
    --noise_mu 0.2 \
    --noise_sigma 0.05