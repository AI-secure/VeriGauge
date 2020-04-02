#!/bin/bash

output_dir=logs/robustness/$(date +"%m%d-%H%M%S")
mkdir -p $output_dir

model_dir=models

for target in random top2 least; do
  for layer in 3 4; do
    for model in ori madry; do
      for alg in recurjac fastlip; do
        if [ $model = "ori" ]; then
          real_model=mnist_${layer}layer_relu_1024
          if [ $layer = "3" ]; then
            eps=0.05
          else
            eps=0.03
          fi
        else
          real_model=mnist_${layer}layer_relu_1024_adv_retrain
          if [ $layer = "3" ]; then
            eps=0.2
          else
            eps=0.1
          fi
        fi
        output=${output_dir}/${real_model}_${alg}_${target}.log
        echo "CUDA_VISIBLE_DEVICES=-1 python3 main.py --numimage 100 --model mnist --eps $eps --norm i --modelfile ${model_dir}/${real_model} --jacbndalg ${alg} --layerbndalg crown-adaptive --lipsteps 30 --targettype ${target} --task robustness > ${output} 2>${output}.err"
      done
    done
  done
done

if [ -z "$(ls -A ${output_dir})" ]; then
  rm -r "${output_dir}"
fi

