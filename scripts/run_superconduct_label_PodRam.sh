#!/bin/sh
#SBATCH -t 24:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=100G
python main.py --dataset0 superconduct --d0_shift_type label --plot_errors True --n_seeds 10 --errs_window 50 --methods fixed_cal none --cs_type abs --schedule variable --bias 1 --init_phase 1 --num_folds 1 --pod_ram_bool True --muh_fun_name NN --num_test_unshifted 0 --test0_size 0.2 --pr_source_delta 0.0005 --pr_target_delta 0.0005 --label_shift 2
