#!/bin/bash

output_dir=logs/landscape/$(date +"%m%d-%H%M%S")
mkdir -p $output_dir

model_dir=models
model_files=mnist_*layer_leaky_20

for model in $model_dir/$model_files; do
  name=$(basename $model)
  for norm in 1 2 i; do
    >&2 output=${output_dir}/${name}_${norm}.log
    # echo "Running $name output $output" 
    echo "CUDA_VISIBLE_DEVICES=-1 python3 main.py --numimage 100 --eps 0.1 --norm $norm --modelfile $model --jacbndalg recurjac --layerbndalg crown-general --lipsteps 30 --task landscape --targettype 1 > $output 2> $output.err"
  done
done

if [ -z "$(ls -A ${output_dir})" ]; then
  echo "Removing ${output_dir}"
  rm -r "${output_dir}"
fi
