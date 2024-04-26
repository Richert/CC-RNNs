#!/bin/bash

# set condition
alphas=( 2.0 4.0 6.0 8.0 10.0 12.0 )
n=50
batch_size=5
range_end=$(($n-1))

# limit amount of threads that each Python process can work with
n_threads=12
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for alpha in "${alphas[@]}"; do
  for IDX in `seq 0 $range_end`; do

    # python calls
    (
    echo "Starting job #$(($IDX+1)) of ${n} jobs for alpha=${alpha}."
    python cluster_rfc_lorenz_alpha.py $alpha $IDX
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
