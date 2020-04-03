#!/bin/bash

"""
run.sh

Run file: collect the samples of gradient norms

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""
# run file for running 1st order and 2nd order clever

if [ "$#" -le 6 ]; then
  echo "Usage: $0 model modeltype nsamp niters activation order target gpuNum"
  echo "model={mnist, cifar}; modeltype={2-layer, normal}; activation={tanh, sigmoid, softplus}; order={1,2}"
  echo "target={top2, rand, least, all}, gpuNum={0,1,2} on frigg, {0,1} on apollo, {0} local"
  exit 1
fi

## set up parameters
# mnist or cifar
model=$1 
# 2-layer, normal(7-layer) 
modeltype=$2
# 200
nsamp=$3
# 10, 20, 50, 100, 200, 300
niters=$4
# tanh, sigmoid, softplus
activation=$5
# 1, 2
order=$6
# 1 (top2), 2 (rand), 4 (least), 15 (all)
target=$7

if [ "$#" -le 7 ]; then
  gpuNum="0"
  #echo "Number of args = $#"
  echo "Using GPU 0"
else
  gpuNum=$8
  #echo "Number of args = $#"
  echo "Using GPU $gpuNum"
fi

if [ "$target" == "top2" ]; then 
  target_type="1"
elif [ "$target" == "rand" ]; then
  target_type="2"
elif [ "$target" == "least" ]; then
  target_type="4"
elif [ "$target" == "all" ]; then
  target_type="15"
else
  echo "Wrong target $target: should be one in {top2, rand, least, all}"
  exit 1
fi

output="${model}_${modeltype}_${activation}_ord${order}_ns${nsamp}_nit${niters}_${target}_$(date +%m%d_%H%M%S)"
dir="logs/$model/$modeltype"
mkdir -p $dir
logfile=$dir/$output.log
echo $logfile

CMD="python3 collect_gradients.py --numimg 20 -d $model --model_name $modeltype --batch_size $nsamp --Nsamps $nsamp --Niters $niters --activation $activation --target_type $target_type --order $order"
echo $CMD

## run on frigg
###CUDA_VISIBLE_DEVICES=$gpuNum
CUDA_VISIBLE_DEVICES=$gpuNum $CMD 2>&1 | tee $logfile
##$CMD 2>&1 | tee $logfile

#NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $CMD 2>&1 | tee $logfile
# NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $CMD >$logfile 2>$logfile.err
echo "Done $logfile"



