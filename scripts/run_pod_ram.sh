#!/bin/bash

python -W ignore main_pod_ram.py \
    --source_conc_type betting \
    --target_conc_type conj-bern \
    --num_of_repeats 10 \
    --eps_tol 0.1 \
    --data_name cifar10 \
    --val_set_size 1000 \
    --corruption_type fog \
    --lr 0.001 \
    --epochs 1 \
    --severity 5 \
    --vis_batch_size 10 \
    --vis_num_of_batches 200