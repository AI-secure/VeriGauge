#!/bin/bash

output_dir=logs/lipschitz/$(date +"%m%d-%H%M%S")
mkdir -p $output_dir

model_files="models/mnist_5layer_tanh_50 models/mnist_7layer_relu_1024 cifar_10layer_relu_2048"

for model in $model_files; do
  name=$(basename $model)
  for alg in recurjac fastlip; do
    for lipsdir in +1 -1; do
      for lipshift in 0 1; do
        nthreads=2
        # for lipsdir == -1, only execute loop once
        if [ $lipsdir = "-1" ] && [ $lipshift = "1" ]; then
          continue
        fi
        if [ $alg = "fastlip" ]; then
          if [ $lipsdir = "+1" ]; then
            continue
          fi
        fi
        output=${output_dir}/${name}_${alg}_${lipsdir}_${lipshift}.log
        echo "CUDA_VISIBLE_DEVICES=-1 python3 main.py --numimage 1 --norm i --modelfile $model --jacbndalg $alg --layerbndalg crown-general --lipsteps 100 --targettype 1 --task lipschitz --liplogstart -3 --liplogend 0 --lipsdir $lipsdir --lipsshift $lipshift --warmup > ${output} 2>${output}.err"
      done
    done
  done
done

if [ -z "$(ls -A ${output_dir})" ]; then
  rm -r "${output_dir}"
fi

