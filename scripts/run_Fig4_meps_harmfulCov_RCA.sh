#!/bin/sh
#SBATCH --partition=parallel
#SBATCH -t 72:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
python ../src/main.py --dataset0 meps --d0_shift_type covariate_harmful --plot_errors True --n_seeds 200 --errs_window 50 --methods fixed_cal_dyn none --cs_type abs --schedule both --bias 3.0 --init_phase 1 --muh_fun_name NN  --num_folds 1 --x_sched_thresh 1000 --num_test_unshifted 500 --test0_size 0.33333 --init_on_test
