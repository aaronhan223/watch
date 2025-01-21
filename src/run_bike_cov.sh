#!/bin/bash

python main.py \
--dataset0 bike_sharing \
--d0_shift_type covariate \
--plot_errors True \
--n_seeds 3 \
--errs_window 50 \
--methods fixed_cal_dyn none \
--cs_type abs \
--schedule variable \
--bias 6 \
--init_phase 1 \
--num_folds 1 \
--muh_fun_name NN \
--x_sched_thresh 1000