#!/bin/sh
#SBATCH --partition=parallel
#SBATCH -t 72:00:00 
#SBATCH --nodes=1 --ntasks=2 --cpus-per-task=2
python ../src/main.py --dataset0 superconduct --d0_shift_type label --plot_errors True --n_seeds 200 --errs_window 50 --methods fixed_cal_dyn none --cs_type abs --schedule both --bias 1 --init_phase 1 --num_folds 1 --muh_fun_name NN --num_test_unshifted 200 --test0_size 0.33333 --label_shift 5 --init_on_test --run_PR_ST --run_PR_CD --pr_cd_batch_size 50
