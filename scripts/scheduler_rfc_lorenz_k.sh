#!/bin/bash

# set condition
ks=( 10 20 40 80 160 320 640 )
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
for k in "${ks[@]}"; do
  for IDX in `seq 0 $range_end`; do

    # python calls
    (
    echo "Starting job #$(($IDX+1)) of ${n} jobs for k=${k}."
    python cluster_rfc_lorenz_k.py $k $IDX
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
