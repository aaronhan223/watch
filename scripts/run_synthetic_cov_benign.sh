#!/bin/bash
python main.py --dataset0 1dim_synthetic_v3 --d0_shift_type covariate --plot_errors True --n_seeds 20 --errs_window 50 --methods fixed_cal_dyn none --cs_type abs --schedule both --bias -0.05 --init_phase 1 --num_folds 1 --muh_fun_name NN --x_ctm_thresh 10 --x_sched_thresh 2000 --num_test_unshifted 500 --test0_size 0.33333 --init_on_test
