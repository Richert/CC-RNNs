#!/bin/bash

# set condition
ks[0]=2
ks[1]=4
ks[2]=8
ks[3]=16
ks[4]=32
ks[5]=64
ks[6]=128
alphas[0]=1.0
alphas[1]=2.0
alphas[2]=4.0
alphas[3]=8.0
alphas[4]=16.0
alphas[5]=32.0
alphas[6]=64.0
n=10
batch_size=4
range_end=$(($n-1))

# limit amount of threads that each Python process can work with
n_threads=10
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for k in ks; do
  for alpha in alphas; do
    for IDX in `seq 0 $range_end`; do

      # python calls
      (
      echo "Starting job #$(($IDX+1)) of ${n} jobs for k=${k} and alpha=${alpha}."
      python cluster_2freq_interpolation.py $IDX $k $alpha
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
