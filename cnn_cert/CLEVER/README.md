**As requested by IBM, this repository is moved to https://github.com/IBM/CLEVER-Robustness-Score, but we aim to keep both repositories synced up.** The code is released under Apache License v2.

CLEVER: A Robustness Metric For Deep Neural Networks
=====================================

CLEVER (**C**ross-**L**ipschitz **E**xtreme **V**alue for n**E**twork **R**obustness) is a metric for
measuring the robustness of deep neural networks.  It estimates the robustness
lower bound by sampling the norm of gradients and fitting a limit distribution
using extreme value theory. CLEVER score is attack-agnostic; a higher score
number indicates that the network is likely to be less venerable to adversarial
examples.  CLEVER can be efficiently computed even for large state-of-the-art
ImageNet models like ResNet-50 and Inception-v3.

For more details, please see our paper:

[Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach](https://openreview.net/pdf?id=BkUHlMZ0b)
by Tsui-Wei Weng\*, Huan Zhang\*, Pin-Yu Chen, Dong Su, Yupeng Gao, Jinfeng Yi, Cho-Jui Hsieh and Luca Daniel

\* Equal contribution

News
-------------------------------------

- Aug 6, 2018: CLEVER evaluation with input transformations (e.g., staircase
  function or JPEG compression) is implemented via BPDA (Backward Pass
  Differentiable Approximation)
- Aug 16, 2018: added 2nd order CLEVER evaluation implementation, which can be
  used to evaluate robustness on classifiers that are twice-differentiable. 

Discussion with Ian Goodfellow and Our Clarifications 
-------------------------------------

We received some inquires on [Ian Goodfellow's
comment](https://arxiv.org/abs/1804.07870) “*Gradient Masking Causes CLEVER to
Overestimate Adversarial Perturbation Size*” on our paper.  We thank Ian for
the discussion but the comments are inappropriate and not applicable to our
paper.  CLEVER is intended to be a tool for network designer and to evaluate
network robustness in the “white-box” setting.  Especially, the argument that
on digital computers all functions are not Lipschitz continuous and behave
like a staircase function (where the gradient is zero almost everywhere) is
incorrect. Under the white-box setting, gradients can be computed via automatic
differentiation, which is well supported by mature packages like TensorFlow.
See [our reply and discussions with Ian Goodfellow on gradient masking and implmentation on digital computers](https://openreview.net/forum?id=BkUHlMZ0b&noteId=Hyc-dnN6f&noteId=SkzxpFrpz).

Setup and train models
-------------------------------------

The code is tested with python3 and TensorFlow v1.3, v1.4 and v1.5. The following
packages are required:

```
sudo apt-get install python3-pip python3-dev
sudo pip3 install --upgrade pip
sudo pip3 install six pillow scipy numpy pandas matplotlib h5py posix_ipc tensorflow-gpu
```

Then clone this repository:

```
git clone https://github.com/huanzhang12/CLEVER.git
cd CLEVER
```

Prepare the MNIST and CIFAR-10 data and models with different activation functions:

```
python3 train_models.py
python3 train_2layer.py
python3 train_nlayer.py --model mnist --modeltype cnn --activation tanh 32 32 64 64 200 200 
python3 train_nlayer.py --model cifar --modeltype cnn --activation tanh 64 64 128 128 256 256 
 
```

To download the ImageNet models:

```
python3 setup_imagenet.py
```

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)


and put the `imgs` folder in `../imagenetdata`, relative to the CLEVER repository. 
This path can be changed in `setup_imagenet.py`.

```
cd ..
mkdir imagenetdata && cd imagenetdata
wget http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz
tar zxf img.tar.gz
cd ../CLEVER
```

How to run
--------------------------------------

### Step 1: Collect gradients

The first step for computing CLEVER score is to collect gradient samples.
The following command collects gradient samples for 10 images in MNIST dataset;
for each image, 3 target attack classes are chosen (random, top-2 and least likely).
Images that are classified incorrectly will be skipped, so you might get less than
10 images.
The default network used has a 7-layer AlexNet-like CNN structure.

```
python3 collect_gradients.py --dataset mnist --numimg 10
```

Results will be saved into folder `lipschitz_mat/mnist_normal` by default (which can be
changed by specifying the `--saved <folder name>` parameter), as
a few `.mat` files.

Run `python3 collect_gradients.py -h` for additional help information.

**Updated:** For model with input transformation, use an additional parameter
`--transform`.  Currently three input transformations are supported (bit-depth
reduction, JPEG compression and PNG compression, corresponding to
`defend_reduce`, `defend_jpeg`, `defend_png` options).  For example:

```
python3 collect_gradients.py --dataset cifar --numimg 10 --transform defend_jpeg
```

You should expect roughly the same CLEVER score with input transformations,
as input transformations do not increase model's intrinsic robustness and can
be broken by BPDA.
See `defense.py` for the implementations of input transformations.

**Updated:** To run 2nd order clever score, run.sh can be used and set order = 2: 

```
./run.sh model modeltype nsamp niters activation order target gpuNum
```
For example, to get 1000 samples of 2nd order clever with 100 iterations on a mnist 7-layer cnn model with tanh activation and random target:

```
./run.sh mnist normal 1000 100 tanh 2 rand 
```

To get samples for the original clever score (1st order approximation), set order = 1. 


### Step 2: Compute the CLEVER score

To compute CLEVER score using the collected gradients, 
run `clever.py` with data saving folder as a parameter:

```
python3 clever.py lipschitz_mat/mnist_normal
```

Run `python3 clever.py -h` for additional help information.


### Step 3: How to interpret the score?

At the end of the output of `clever.py`, you will see three `[STATS][L0]` lines similar to the following:

```
[STATS][L0] info = least, least_clever_L1 = 2.7518, least_clever_L2 = 1.1374, least_clever_Li = 0.080179
[STATS][L0] info = random, random_clever_L1 = 2.9561, random_clever_L2 = 1.1213, random_clever_Li = 0.075569
[STATS][L0] info = top2, top2_clever_L1 = 1.6683, top2_clever_L2 = 0.70122, top2_clever_Li = 0.050181
```

The scores shown are the average scores for all (in the example above, 10)
images, with three different target attack classes: least likely, random and
top-2 (the class with second largest probability).  Three scores are provided:
CLEVER\_L2, CLEVER\_Linf and CLEVER\_L1, representing the robustness for L2, L\_infinity
and L1 perturbations. CLEVER score for Lp norm roughly reflects
the minimum Lp norm of adversarial perturbations. A higher CLEVER score
indicates better network robustness, as the minimum adversarial perturbation is
likely to have a larger Lp norm. As CLEVER uses a sampling based method, the 
scores may vary slightly for different runs.

More Examples
---------------------------------

For example, the following command will evaluate the CLEVER scores on 1
ImageNet image, for a 50-layer ResNet model. We set the number of gradient
samples per iterations to 512, and run 100 iterations:

```
python3 collect_gradients.py --dataset imagenet --model_name resnet_v2_50 -N 512 -i 100
python3 clever.py lipschitz_mat/imagenet_resnet_v2_50/
```
<p align="center">
  <img src="http://www.huan-zhang.com/images/upload/clever/139.00029510.jpg" alt="Bustard"/>
</p>

For this image (`139.00029510.jpg`, which is the first image given the default
random seed) in dataset, the original class is 139 (bustard), least likely
class is 20 (chickadee), top-2 class is 82 (ptarmigan), random class target
is 708 (pay-phone).  (These can be observed in `[DATAGEN][L1]` lines of the
output of `collect_gradients.py`). We get the following CLEVER scores:

```
[STATS][L0] info = least, least_clever_L1 = 8.1393, least_clever_L2 = 0.64424, least_clever_Li = 0.0029474 
[STATS][L0] info = random, random_clever_L1 = 4.6543, random_clever_L2 = 0.61181, random_clever_Li = 0.0023765 
[STATS][L0] info = top2, top2_clever_L1 = 0.99283, top2_clever_L2 = 0.13185, top2_clever_Li = 0.00062238
```

The L2 CLEVER score for the top-2, random and least-likely classes are 
0.13185, 0.61181 and 0.64424,
respectively.  It indicates that it is very easy to attack this image from
class 139 to 82.  We then run the CW attack, which is the strongest L2 attack
to date, on this image with the same three target classes. The distortion of
adversarial images are 0.1598, 0.82025, 0.85298 for the three targets.
Indeed, to misclassify the image to class 82, only a very small distortion
(0.1598) is needed. Also, the CLEVER scores are (usually) less than the L2
distortions observed on adversarial examples, but are not too small to be
useless, reflecting the nature that CLEVER is an estimated robustness lower
bound.

CLEVER also has an untargeted version, which is essentially the smallest CLEVER
score over all possible target classes. The following examples shows how to
compute untargeted CLEVER score for 10 images from MNIST dataset, on the
2-layer MLP model:

```
python3 collect_gradients.py --data mnist --model_name 2-layer --target_type 16 --numimg 10
python3 clever.py --untargeted ./lipschitz_mat/mnist_2-layer/
```

Target type 16 (bit 4 set to 1) indicates that we are collecting gradients for
untargeted CLEVER score (see `python3 collect_gradients.py -h` for more details).
The results will look like the following:

```
[STATS][L0] info = untargeted, untargeted_clever_L1 = 3.4482, untargeted_clever_L2 = 0.69393, untargeted_clever_Li = 0.035387
```

For datasets which have many classes, it is very expensive to evaluate the
untargeted CLEVER scores.  However, usually the robustness of the top-2
targeted class can roughly reflect the untargeted robustness, as it is
usually one of the easiest classes to change to.

Built-in Models 
-------------------------------- 

In the examples shown above we have used several different models.
The code on this repository has a large number of built-in models for
robustness evaluation.  Model can be selected by changing the `--model_name`
parameter to `collect_gradiets.py`.  For MNIST and CIFAR dataset, the following
models are available: "2-layer" (MLP), "normal" (7-layer CNN), "distilled"
(7-layer CNN with defensive distillation), "brelu" (7-layer CNN with Bounded
ReLU).  For ImageNet, available options are: "resnet_v2_50", "resnet_v2_101",
"resnet_v2_152", "inception_v1", "inception_v2", "inception_v3",
"inception_v4", "inception_resnet_v2", "vgg_16", "vgg_19", "mobilenet_v1_025",
"mobilenet_v1_050", "mobilenet_v1_100", "alexnet", "densenet121_k32",
"densenet169_k32", "densenet161_k48" and "nasnet_larget".
A total of 18 ImageNet models have been built in so far.


How to evaluate my own model?
--------------------------------

Models for MNIST, CIFAR and ImageNet datasets are defined in `setup_mnist.py`,
`setup_cifar.py` and `setup_imagenet.py`. For MNIST and CIFAR, you can modify
the model definition in `setup_mnist.py` and `setup_cifar.py` directly.  For
ImageNet, a protobuf (.pb) model with frozen network parameters is expected,
and new ImageNet models can be added into `setup_imagenet.py` by adding a new
`AddModel()` entry, similar to other ImageNet models. Please read the comments 
on `AddModel()` in `setup_imagenet.py` for more details.

The following two links provide examples on how to prepare a frozen protobuf
for ImageNet models:

[Prepare DenseNet models](https://github.com/huanzhang12/tensorflow-densenet-models)

[Prepare AlexNet model](https://github.com/huanzhang12/tensorflow-alexnet-model)

Known Issues
--------------------------------

If you encounter the following error:

```
posix_ipc.ExistentialError: Shared memory with the specified name already exists
```

Please delete those residual files in `/dev/shm`

```
rm -f /dev/shm/*all_inputs
rm -f /dev/shm/*input_example
rm -f /dev/shm/*randsphere
rm -f /dev/shm/*scale
```

For systemd based Linux distributions (for example, Ubuntu 16.04+), it is
necessary to set `RemoveIPC=no` in `/etc/systemd/logind.conf` and restart
`systemd-logind` (`sudo systemctl restart systemd-logind.service`) to avoid
systemd from removing shared memory objects after user logout (which prevents
CLEVER running in background).

