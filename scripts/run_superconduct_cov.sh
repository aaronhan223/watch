#!/bin/sh
#SBATCH -t 24:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=100G
python main.py --dataset0 superconduct --d0_shift_type covariate --plot_errors True --n_seeds 100 --errs_window 50 --methods fixed_cal_dyn none --cs_type abs --schedule variable --bias 2.0 --init_phase 1 --num_folds 1 --muh_fun_name NN --x_sched_thresh 2000 --num_test_unshifted 1000 --test0_size 0.33333
