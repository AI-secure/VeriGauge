Reference Implementation of CROWN, RecurJac, Fast-Lin and Fast-Lip
=====================================

This repository contains a Numpy reference implementation of CROWN, RecurJac,
Fast-Lin and Fast-Lip.

A new **PyTorch** implementation of CROWN and Fast-Lin can be found in the
[CROWN-IBP repository](https://github.com/huanzhang12/CROWN-IBP#compute-crown-verified-errors), which
includes implementation of **CNN** layers and efficient computation on **GPUs**.

Intro
===========

**CROWN** is a neural network verification framework that linearizes general
activation functions and creates linear upper and lower bounds for neural
network outputs in a layer-by-layer manner. CROWN can be seen as a special case
of solving a LP relaxed neural network (Salman et al., 2019), but is much more
efficient than LP solvers. CROWN can be used as a general tool for bounding
neural networks. It generalizes and outperforms our previous algorithm
(Fast-Lin) by giving tighter verification bounds for ReLU networks as well as
supporting other general activation functions.

**RecurJac** is an efficient algorithm for obtaining element-wise lower and
upper bounds of Jacobian matrix with respect to a neural network's input, and
can be used for computing local or global Lipschitz constants, as well as
neural network robustness verification.  RecurJac significantly improves the
quality of Lipschitz constant obtained by our earlier algorithm, Fast-Lip.
The improvement comes from a recursive refinement of the layer-wise Jacobian
bound.  Additionally, RecurJac applies to a wide range of activation functions
under mild assumptions (Fast-Lip is for ReLU only).

This repository contains many improvements, cleanups and fixes to the
implementation of our previous algorithms (Fast-Lin, Fast-Lip and CROWN).  **It
is suggested to use this repository as a reference code base for all four
algorithms (RecurJac, CROWN, Fast-Lin and Fast-Lip)**. 

More details for our algorithms can be found in the following papers:


[1] **RecurJac: An Efficient Recursive Algorithm for Bounding Jacobian Matrix of Neural Networks and Its Applications**, 
Huan Zhang, Pengchuan Zhang, Cho-Jui Hsieh, *AAAI 2019* ([PDF](https://arxiv.org/pdf/1810.11783.pdf)) ([Slides](http://www.huan-zhang.com/pdf/RecurJac_Slides.pdf))


[2] **Efficient Neural Network Robustness Certification with General Activation Functions**, 
Huan Zhang\*,Tsui-Wei Weng\*, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel,
*NIPS 2018* ([PDF](http://arxiv.org/pdf/1811.00866.pdf))


[3] **Towards Fast Computation of Certified Robustness for ReLU Networks**, 
Tsui-Wei Weng\*, Huan Zhang\*, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane Boning, Inderjit S. Dhillon, Luca Daniel,
*ICML 2018* ([PDF](https://arxiv.org/abs/1804.09699))

\* indicates equal contribution. BibTex citations can be found at the end of this document.



Installation Instructions
-----------------------

Our code is compatible with python3 and TensorFlow v1.8 to v1.11. We recommend
using conda as the Python package manager.  Our code requires the following Conda packages:

```
conda install pillow numpy scipy pandas tensorflow-gpu keras-gpu h5py pillow
conda install --channel numba llvmlite numba
grep 'AMD' /proc/cpuinfo >/dev/null && conda install nomkl
```

If you system does not have GPU, you can replace `tensorflow-gpu` and
`keras-gpu` with package `tensorflow` and `keras`, which provides the CPU
version. Using GPU is not necessary for running experiments, as our main
procedures are implemented using numpy. The third command is required on AMD
systems because the default numpy package links to Intel MKL which can run
quite slow on AMD CPUs.

After installing prerequisites, clone our repository:

```
git clone https://github.com/huanzhang12/RecurJac-and-CROWN.git
cd RecurJac-and-CROWN
```

RecurJac pretrained models can be download here:

```
wget http://download.huan-zhang.com/models/adv/robustness/models_recurjac.tar
tar xvf models_recurjac.tar
```

For models used in CROWN (paper [2]):
```
wget http://download.huan-zhang.com/models/adv/robustness/models_crown.tar
tar xvf models_crown.tar
```

For models used in Fast-Lin/Fast-Lip (paper [1]):
```
wget http://download.huan-zhang.com/models/adv/robustness/models_relu_verification.tar
tar xvf models_relu_verification.tar
```

This will create a `models` folder, containing a few Keras model files used in our papers.

How to Run
--------------------

The main interface for computing bounds is `main.py`. We first give some examples on how
to use it. Note that the first run may take some time because any necessary datasets will
be downloaded automatically.

### Robustness Certification (RecurJac, CROWN, Fast-Lin, Fast-Lip)

In this task, we evaluate the robustness lower bound of a neural network model.
Within this lower bound, we can guarantee that no adversarial examples exist.
This task is reported in Table 1 in our paper [1].

Using the following command, we run RecurJac on *10* correctly classified
images, and compute the robustness lower bound against a *random* target class,
using the model file *`models/mnist_3layer_relu_1024_adv_retrain`* (a 3-layer
MNIST model with adversarial training). We use *CROWN-adaptive* to compute
layer-wise outer bounds and use *RecurJac* to compute robustness certification.
The input is perturb in a *L_inf* ball.  The initial perturbation is set to
*0.2*, and a binary search procedure finds the best lower bound (so the initial
value is not very important).

```
python3 main.py --task robustness --numimage 10 --targettype random --norm i --modelfile models/mnist_3layer_relu_1024_adv_retrain --layerbndalg crown-adaptive --jacbndalg recurjac --eps 0.2
```

The last line of output is something like:

```
[L0] model = models/mnist_3layer_relu_1024_adv_retrain, avg robustness_lb = 0.14417, numimage = 10
```

which indicates the average robustness lower bound for 10 images is 0.14417.

For CROWN and Fast-Lin, `--jacbndalg` should be set to `disabled`, and
`--layerbndalg` should be set to `crown-adaptive` or `fastlin` for ReLU networks,
or `crown-general` for other activation functions. To favor running time
instead of getting the tightest bound, you can decrease `--lipstep` and
`--step` parameters to reduce the number of integral intervals and binary
search steps (the defaults are 15 and 15, respectively).


### Lipschitz Constant Computation (RecurJac, Fast-Lip)

In this task, we evaluate the local Lipschitz constant within different radii
*eps* inside a Lp ball.  This task is reported in Figure 3 in our paper [1].

Using the following command, we run RecurJac on *1* correctly classified image
from class *1*, using the model file *`models/mnist_7layer_relu_1024`*. We use
*CROWN-adaptive* to compute layer-wise outer bounds and use *RecurJac* to
compute the Jacobian bounds. The input is perturb in a *L_inf* ball. We
evaluate *20* eps values, starting from *10e-3*=0.001 to *10e0*=1.0, in a
logarithmic scale. 

```
python3 main.py --task lipschitz --numimage 1 --targettype 1 --modelfile models/mnist_7layer_relu_1024 --layerbndalg crown-adaptive --jacbndalg recurjac --norm i --lipsteps 20 --liplogstart -3 --liplogend 0
```

At the last line of output, we obtain the following results:

``` 
[L0] model = models/mnist_7layer_relu_1024, numimage = 1,
avg_lipschitz[0.00000] = 172.04774, avg_lipschitz[0.00100] = 172.15767,
avg_lipschitz[0.00144] = 176.57712, avg_lipschitz[0.00207] = 178.59306,
avg_lipschitz[0.00298] = 193.93967, avg_lipschitz[0.00428] = 209.47284,
avg_lipschitz[0.00616] = 252.63580, avg_lipschitz[0.00886] = 319.40561,
avg_lipschitz[0.01274] = 486.35843, avg_lipschitz[0.01833] = 2157.19458,
avg_lipschitz[0.02637] = 368216.12500, avg_lipschitz[0.03793] = 917739.12500,
avg_lipschitz[0.05456] = 1293761.00000, avg_lipschitz[0.07848] = 1592493.62500,
avg_lipschitz[0.11288] = 1881139.37500, avg_lipschitz[0.16238] = 2012919.00000,
avg_lipschitz[0.23357] = 2024336.25000, avg_lipschitz[0.33598] = 2026857.75000,
avg_lipschitz[0.48329] = 2027686.12500, avg_lipschitz[0.69519] = 2027717.50000,
avg_lipschitz[1.00000] = 2027717.50000, opnorm_global_lipschitz = 19351028.5323
```

When eps = 0, the Lipschitz constant is the same as the norm of gradient at
input x.  When eps increases, local Lipschitz constant also increases.
Eventually, the local Lipschitz constants saturate at 2027717.5 regardless of
the increasing eps.  This is thus a global Lipschitz constant, and is much
better than the value obtained by the product of operator norm (19351028.5323).

Changing `--jacbndalg` parameter value to `fastlip to run the same task using Fast-Lip in
our paper [3].


### Local Optimization Landscape (RecurJac)

In this task, we try to find the largest perturbation (denoted as R\_max in
some Lp norm) where at least one element in Jacobian are knwon to be positive
or negative. In other words, no points have a zero gradient inside this Lp ball
with radius R\_max. This tasks is reported in Figure 2 in our paper [1].

Using the following command, we run RecurJac on *100* correctly classified
images from class *1*, using the model file *`models/mnist_5layer_leaky_20`* (a
5 layer leaky-ReLU MNIST network with 20 neurons per layer).  We use
*CROWN-general* to compute layer-wise outer bounds and use *RecurJac* to
compute the Jacobian bounds.  The input is perturb in a *L2* ball. The initial
perturbation is set to *0.1*, and a binary search procedure is used to find
R\_max (so the initial value is not very important).

```
python3 main.py --task landscape --numimage 100 --targettype 1 --modelfile models/mnist_5layer_leaky_20 --layerbndalg crown-general --jacbndalg recurjac --norm 2 --eps 0.1 --lipsteps 1
```

If `lipsteps` is set to values other than 1, some additional values of eps will
be evaluated between 0 and R\_max, which is not necessary for this task.

The command above will print results to the terminal. The final results are at
the last line:

```
[L0] model = models/mnist_5layer_leaky_20, numimage = 100, avg_max_eps = 0.21103, avg_lipschitz_max = 27.2158, opnorm_global_lipschitz = 177.2929
```

where `avg_max_eps` is the average R\_max over 100 images.


Train your own model
--------------------

Our current implementation can load any model files created by
`train_nlayer.py`.  For example, the following command trains a network with 2
hidden layers, with 20 neurons per hidden layer, using leaky-ReLU activation
function with a negative side slope 0.3.  Model will be saved to the current
folder `.`:

```
python train_nlayer.py --activation leaky --leaky_slope 0.3 --modelpath . 20 20
```

Run `python train_nlayer.py -h` to see a list of tunable parameters (learning
rate, weight decay, activation function, etc). Feel free to edit `train_nlayer.py`
to add your tricks to make the network more robust, and verify robustness
with CROWN or RecurJac.

Usage
--------------------

This section lists all command-line arguments for `main.py`.

```
  --modelfile MODELFILE
                        Path to a Keras model file. See train_nlayer.py for
                        training a compatible model file.
  --dataset {auto,mnist,cifar}
                        Dataset to be used. When set to "auto", it will
                        automatically detect dataset based on the input
                        dimension of model file.
  --task TASK           Define a task to run. This will call the "task_*.py"
                        files under this directory. Currently supported tasks:
                        robustness, landscape, lipschitz.
  --numimage NUMIMAGE   Number of images to run.
  --startimage STARTIMAGE
                        First image index in dataset.
  --norm {i,1,2}        Perturbation norm: "i": Linf, "1": L1, "2": L2.
  --layerbndalg {crown-general,crown-adaptive,fastlin,spectral}
                        Algorithm to compute layer-wise upper and lower bounds. "crown-general": CROWN for
                        general activation functions, "crown-adaptive": CROWN
                        for ReLU with adaptive upper and lower bounds,
                        "fastlin": Fast-Lin, "spectral": spectral norm bounds
                        (special, when use "spectral" bound we simply multiply
                        each layer's operator norm).
  --jacbndalg {disable,recurjac,fastlip}
                        Algorithm to compute Jacobian bounds. Used to compute (local)
                        Lipschitz constant and robustness verification. When
                        set to "disable", --layerbndalg will be used to compute
                        robustness lower bound.
  --lipsdir {-1,1}      RecurJac bounding order, -1 backward, +1 forward.
                        Usually set to -1.
  --lipsshift {0,1}     Shift RecurJac forward pass bounding by 1 layer (i.e.,
                        starting from layer 2 rather than layer 1; useful when
                        the input layer has a large number of neurons).
                        Usually set to 0.
  --lipsteps LIPSTEPS   Task specific. For the "lipschitz" task, this
                        parameter specifies the number of eps values to
                        evaluate local Lipschitz constants. For the
                        "robustness" task, this parameter is the number of
                        intervals for numerical integration; a larger value
                        gives a better bound.
  --eps EPS             Inital epsilon for "landscape" and "robustness" tasks.
  --liplogstart LIPLOGSTART
                        Only used in "lipschitz" task. When LIPLOGSTART !=
                        LIPLOGSEND, we generate epsilon between LIPLOGSTART
                        and LIPLOGEND using np.logspace with LIPSTEPS steps.
  --liplogend LIPLOGEND
                        See --liplogstart
  --quad                Use quadratic bound (for 2-layer ReLU network, CROWN-
                        adaptive only).
  --warmup              Warm up before start timing. The first run will
                        compile python code to native code, which takes a
                        relatively long time, and should be excluded from
                        timing.
  --targettype TARGETTYPE
                        Target class label for robustness verification. Can be
                        "least", "runnerup", "random" or "untargeted", or a
                        number to specify class label.
  --steps STEPS         Number of steps to do binary search.
  --seed SEED           Random seed.
```

BibTex Entries
--------------------------------

```
@inproceedings{zhang2018recurjac,
  author = "Huan Zhang AND Pengchuan Zhang AND Cho-Jui Hsieh",
  title = "RecurJac: An Efficient Recursive Algorithm for Bounding Jacobian Matrix of Neural Networks and Its Applications",
  booktitle = "AAAI Conference on Artificial Intelligence (AAAI), arXiv preprint arXiv:1810.11783",
  year = "2019",
  month = "dec"
}
```

```
@inproceedings{zhang2018crown,
  author = "Huan Zhang AND Tsui-Wei Weng AND Pin-Yu Chen AND Cho-Jui Hsieh AND Luca Daniel",
  title = "Efficient Neural Network Robustness Certification with General Activation Functions",
  booktitle = "Advances in Neural Information Processing Systems (NIPS), arXiv preprint arXiv:1811.00866",
  year = "2018",
  month = "dec"
}
```

```
@inproceedings{salman2019convex,
  author={Salman, Hadi and Yang, Greg and Zhang, Huan and Hsieh, Cho-Jui and Zhang, Pengchuan},
  title={A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks},
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS), arXiv preprint arXiv:1902.08722",
  year= "2019",
  month = "dec"
}
```

```
@inproceedings{weng2018towardsfa,
  author = "Tsui-Wei Weng AND Huan Zhang AND Hongge Chen AND Zhao Song AND Cho-Jui Hsieh AND Duane Boning AND Inderjit S. Dhillon AND Luca Daniel",
  title = "Towards Fast Computation of Certified Robustness for ReLU Networks",
  booktitle = "International Conference on Machine Learning (ICML), arXiv preprint arXiv:1804.09699",
  page = "5273-5282",
  year = "2018",
  month = "jul"
}
```


