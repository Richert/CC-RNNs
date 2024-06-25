#!/bin/bash

# set condition
delay=( 5 10 15 20 25 30 )
noise=( 0.0 0.1 0.2 0.4 0.8 1.6 )
n=50
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
for d in "${delay[@]}"; do
  for lvl in "${noise[@]}"; do
    for IDX in `seq 0 $range_end`; do

      # python calls
      (
      echo "Starting job #$(($IDX+1)) of ${n} jobs for noise level = ${lvl} and delay = ${d}."
      python cluster_lr_delayedchoice.py $d $lvl $IDX
      sleep 1
      ) &

      # batch control
      if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
            wait -n
      fi

    done
  done
done

wait
echo "All jobs finished."
