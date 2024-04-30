#!/bin/bash

# set condition
steps=( 50000 100000 150000 200000 250000 300000 350000 400000 450000 500000 )
n=20
batch_size=8
range_end=$(($n-1))

# limit amount of threads that each Python process can work with
n_threads=10
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for step in "${steps[@]}"; do
  for IDX in `seq 0 $range_end`; do

    # python calls
    (
    echo "Starting job #$(($IDX+1)) of ${n} jobs for number of training steps = ${step}."
    python cluster_rfc_lorenz.py $step $IDX
    python cluster_lr_lorenz.py $step $IDX
    python cluster_clr_lorenz.py $step $IDX
    sleep 1
    ) &

    # batch control
    if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
          wait -n
    fi

  done
done

wait
echo "All jobs finished."
