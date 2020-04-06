export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

set -e

# Test for RecurJac on Landscape
numactl -N 0 python3 main.py --numimage 1 --eps 0.1 --norm 2 --modelfile models/mnist_5layer_leaky_20 --jacbndalg recurjac --layerbndalg crown-general --lipsteps 5 --task landscape --targettype 1

# Test for RecurJac on Lipschitz constant
numactl -N 0 python3 main.py --numimage 1 --norm i --modelfile models/mnist_5layer_tanh_50 --jacbndalg recurjac --layerbndalg crown-general --lipsteps 5 --targettype 1 --task lipschitz --liplogstart -3 --liplogend 0 --lipsdir -1

export NUMBA_DISABLE_JIT=0
# Test for RecurJac on Robustness
numactl -N 0 python3 main.py --numimage 1 --eps 0.05 --norm i --modelfile models/mnist_3layer_relu_1024_adv_retrain --jacbndalg recurjac --layerbndalg crown-general --lipsteps 10 --targettype least --task robustness

# Test for CROWN-adaptive
numactl -N 0 python3 main.py --numimage 1 --eps 0.05 --norm i --modelfile models/mnist_3layer_relu_1024_best --jacbndalg disable --layerbndalg crown-adaptive --targettype least --task robustness

# Test for CROWN-general
numactl -N 0 python3 main.py --numimage 1 --eps 0.5 --norm 1 --modelfile models/mnist_4layer_arctan_1024 --jacbndalg disable --layerbndalg crown-general --targettype least --task robustness

# Test for CROWN-adaptive with the second layer using quadratic bound
numactl -N 0 python3 main.py --numimage 1 --eps 0.5 --norm i --modelfile models/mnist_3layer_relu_20_best --jacbndalg disable --layerbndalg crown-general --quad --targettype least --task robustness

# Test for fastlin
numactl -N 0 python3 main.py --numimage 1 --eps 0.05 --norm i --modelfile models/mnist_3layer_relu_1024_best --jacbndalg disable --layerbndalg fastlin --targettype least --task robustness

# Test for Fast-Lip
numactl -N 0 python3 main.py --numimage 1 --eps 0.05 --norm i --modelfile models/mnist_3layer_relu_1024_adv_retrain --jacbndalg fastlip --layerbndalg fastlin --lipsteps 10 --targettype least --task robustness

