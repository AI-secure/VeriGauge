# VeriGauge: Unified Toolbox for Representative Robustness Verification Approaches for Deep Neural Networks

llylly (linyi2@illinois.edu, [linyil.com](http://linyil.com/)) @ Secure Learning Lab, UIUC



This is a unified toolbox for representative robustness verification approaches for DNNs. The leader board for different approaches can be found here: https://github.com/AI-secure/Provable-Training-and-Verification-Approaches-Towards-Robust-Neural-Networks.

Related paper: "SoK: Certified Robustness for Deep Neural Networks".

- What is robustness verification for DNNs?

  DNNs are vulnerable to adversarial examples. Given a model and an input x0, the robustness verification approaches can certify that there are no adversarial samples around x0 within radius r. The complete verification of DNNs is NP-complete [1,2]. Therefore, current verification approaches usually leverage relaxations, which results in outputting smaller r than the real one.

- What neural networks are supported?

  Currently, existing approaches mainly support image classification tasks for MNIST, CIFAR-10, and ImageNet, and our toolbox supports all of them though networks for ImageNet are usually too large for standard non-probabilistic verification approaches. Though a significant amount of verification approaches support skip connections, max-pooling layers, etc, typical verification approaches mainly work on *feed-forward neural networks with ReLU activations*, containing only fully-connected layers and convolutional layers.

**Main Features:**

1. A unified lightweight platform for running about 20 verification approaches in a simple PyTorch-based interface.
2. Easily extensible to your own customized neural networks.
3. Easily extensible to your own verification approaches.
4. High-efficiency benefited from the lightweight structure.



[1] Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. https://arxiv.org/abs/1702.01135.

[2] Towards Fast Computation of Certified Robustness for ReLU Networks. http://proceedings.mlr.press/v80/weng18a.html.



## Supported Approach List

| Approach Type      | ClassName            | Description & Path                                           | Comments                                                   |
| ------------------ | -------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| *Normal Infernace* | CleanAdaptor         | The normal inference of the model. Implemented in `adaptor.basic_adaptor`. | Not a verification approach.                               |
| *Empirical Attack* | PGDAdaptor           | Based on Python toolbox `cleverhans`. Implemented in `adaptor.basic_adaptor`. | Not a verification approach, just provide upper bound of r |
| *Empirical Attack* | CWAdaptor            | Based on Python toolbox `cleverhans`. Implemented in `adaptor.basic_adaptor`. | Not a verification approach, just provide upper bound of r |
| MILP               | MILPAdaptor          | Reimplementation of [Tjeng et al's](https://arxiv.org/abs/1711.07356) MILP-based verification based on Python's Gurobi API. Adaptor in `adaptor.basic_adaptor`. Core in `basic.fastmilp.MILPVerifier`. | Complete Verification                                      |
| SDP                | PercySDPAdaptor      | Reimplementation of [Raghunathan et al's](https://arxiv.org/abs/1811.01057) SDP-based verification based on `cvxpy`. Adaptor in `adaptor.basic_adaptor`. Core in `basic.percysdp`. |                                                            |
| SDP                | FazlybSDPAdaptor     | Reimplementation of [Fazlyb et al's](https://arxiv.org/abs/1903.01287) SDP-based verification based on `cvxpy`. Adaptor in `adaptor.basic_adaptor`. Core in `basic.components.BaselinePointVerifierExt`. |                                                            |
| Linear-Based       | FastLinIBPAdaptor    | Reimplementation of the combination of IBP and FastLin bound ($l$ and $u$ per layer are the maximum or minimum of two bounds respectively). Adaptor in `adaptor.basic_adaptor`. Core in `basic.intervalbound`. |                                                            |
| Linear-Based       | FastLinAdaptor       | [Weng et al's](http://proceedings.mlr.press/v80/weng18a.html) linear bound propagation based verification approach. Adaptor in `adaptor.recurjac_adaptor`. Core in `recurjac/`. |                                                            |
| Linear-Based       | FastLinSparseAdaptor | [Weng et al's](http://proceedings.mlr.press/v80/weng18a.html) linear bound propagation based verification approach. This implementation is accelerated by sparse matrix multiplication. Adaptor in `adaptor.cnncert_adaptor`. Core in `cnn_cert/`. |                                                            |
| Linear-Based       | CNNCertAdaptor       | [Boopathy et al's](https://arxiv.org/abs/1811.12395) linear bound propagation based  verification approach. This approach extends FastLin and CROWN to general CNN/Residual/Sigmoid neural networks with high efficiency. Adaptor in `adaptor.cnncert_adaptor`. Core in `cnn_cert/`. |                                                            |
| Linear-Based       | LPAllAdaptor         | LP-full verification approach, which computes $l$ and $u$ layerwise by linear programming. It is mentioned by [Boopathy et al](https://arxiv.org/abs/1811.12395), [Weng et al](http://proceedings.mlr.press/v80/weng18a.html), and analyzed by [Salman et al](https://arxiv.org/abs/1902.08722). Here, the adaptor is in `adaptor.cnncert_adaptor`. Core in `cnn_cert/` (we use Boopathy et al's implementation). |                                                            |
| Linear-Based       | ZicoDualAdaptor      | [Wong et al's](http://arxiv.org/abs/1711.00851) linear dual-based verification approach. Adaptor in `adaptor.lpdual_adaptor`. Core in `convex_adversarial/`. |                                                            |
| Linear-Based       | FullCrownAdaptor     | [Zhang et al's](https://arxiv.org/abs/1811.00866) linear bound propagation based verification approach. Adaptor in `adaptor.crown_adaptor`. Core in `crown_ibp/`. |                                                            |
| Linear-Based       | CrownIBPAdaptor      | [Zhang et al's](https://arxiv.org/abs/1906.06316) linear + interval bound propagation based verification approach. Adaptor in `adaptor.crown_adaptor`. Core in `crown_ibp/`. |                                                            |
| Linear-Based       | IBPAdaptor           | [Gowal et al's](https://arxiv.org/abs/1810.12715) interval propagation based verification approach. Adaptor in `adaptor.crown_adaptor`. Core in `crown_ibp/`. The re-implementation of our own is available at `adaptor.basic_adaptor.IBPAdaptor`, which has similar performance as Zhang et al's and Gowal et al's implementation. For simplicity, by default it uses Zhang et al's implementation. |                                                            |
| Lipschitz          | FastLipAdaptor       | [Weng et al's](http://proceedings.mlr.press/v80/weng18a.html) Lipschitz based verification approach. Adaptor in `adaptor.recurjac_adaptor`. Core in `recurjac/`. |                                                            |
| Lipschitz          | RecurJacAdaptor      | [Zhang et al's](https://arxiv.org/abs/1810.11783) Lipschitz based verification approach. Adaptor in `adaptor.recurjac_adaptor`. Core in `recurjac/`. |                                                            |
| Lipschitz          | SpectralAdaptor      | [Szegedy et al's](https://arxiv.org/abs/1312.6199) Lipschitz (spectral bound) based verification approach. Adaptor in `adaptor.recurjac_adaptor`. Core in `recurjac/` (we leverage Zhang et al's implementation). |                                                            |
| Branch and Bound   | AI2Adaptor           | [Gehr et al's](https://www.cs.rice.edu/~sc40/pubs/ai2.pdf) branch-and-bound based complete verification approach (concretely, domain of set of polyhedra). Adaptor in `adatpro.eran_adaptor`. Core in `eran/`. | Complete Verification                                      |
| Linear-Based       | DeepPolyAdaptor      | [Singh et al's](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf) linear relaxation-based verification approach. Adaptor in `adatpro.eran_adaptor`. Core in `eran/`. |                                                            |
| Hybrid             | RefineZonoAdaptor    | [Singh et al's](https://files.sri.inf.ethz.ch/website/papers/RefineZono.pdf) linear relaxation + MILP + IBP hybrid verification approach. Adaptor in `adatpro.eran_adaptor`. Core in `eran/`. |                                                            |
| Linear-Based       | KReluAdaptor         | [Singh et al's](https://papers.nips.cc/paper/9646-beyond-the-single-neuron-convex-barrier-for-neural-network-certification) linear relaxation based verification approach with $l$ and $u$ refinement from multiple neuron's relaxation. Adaptor in `adatpro.eran_adaptor`. Core in `eran/`. |                                                            |





## Prerequisites

1. Find a server with GPU support and Linux / MacOS system. (The toolbox has been tested on Linux and MacOS. It should be possible to run on Windows, but we can't guarantee so.)

2. Prepare necessary datasets:

   1. If you only want to benchmark on MNIST and CIFAR-10, don't need to do anything, since PyTorch will automatically download them later.

   2. If you want to benchmark on ImageNet, in `datasets.py`, please set the Line 12 to the path of ImageNet dataset on your local environment. The dataset should be organized according to the instruction above Line 12.

3. Install necessary packages according to `requirements.txt`.

4. Download model weights at: https://drive.google.com/drive/folders/1vh7dwvn1P544r5rzOfJFPzShV4-UsTAv?usp=sharing, then store the whole folder as `models_weights/exp_models/`.  Or if you like, you can also train and load your own models.

5. Set your global Keras settings in `~/.keras/keras.json` as below.

```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_first"
}
```

6. Install ELINA, and DeepG according to the instructions in `eran/install.sh`. Then specify their path in `constants.py` (Line 5 and Line 6).

7.  Create an empty folder named `tmp/` under the tool root path.

Then you are all set!



## Main Usage

#### Replicate Our Results in The Paper

The `experiments/run.sh` and `experiments/run_cifar.sh` contain concrete commands for the users to replicate our results.

In the arguments of these commands, `--method` specifies the verification approaches to run, ``--dataset`` specifies the dataset to run, `--model` specifies the model to run, ``--weight`` specifies the model weight to run, and `--mode` specifies the type of evaluation to run. All of them supports multiple arguments, and multiple settings will be ran sequentially.

- `--method`: From `experiments/data_analyzer`, the dictionary `approach_mapper` defines the name mapping between paper's verification approach names to argument names.
- `--dataset`: It should be names in `contants.py`'s `DATASETS` list.
- `--model`: ranging from A to G, where A = FCNNa, B = FCNNb, C = CNNa, D = CNNb, E = CNNc, F = CNNd, and G = FCNNc.
- `--weight`: same names as trained weight settings in the paper.
- `--mode`: should be either "verify" - compute the robust accuracy, or "radius" - compute the average certified robustness radius.



#### Verify Your Own Models

The main script `main.py` shows how to run the toolkit on your customized tasks.

To enable the support for new verification approaches, you could copy the implementation folder to the project root folder, then write your own adaptor following the `adaptor.adaptor.Adaptor` template, and strore it in `adaptor/` folder.

To enable the support for your own model, you could write your model loading function in `models/` (the function should load both structure and weights, and should be PyTorch model), and register the new model in `model.py`.

- If the model works on normalized input, you should treat the input normalization as the additional first layer of your model, and this first layer could be implemented by `datasets.NormalizeLayer`.
- If the model contains flatten layer, it should be replaced by our flatten layer implementation in `models.zoo.Flatten`.

To enable the support for new datasets, you could extend `datasets.py`, and register the dataset name in `constants.py`.



## Base Repos and Tool Structure Overview

Our toolbox is based on following tools, which are stored in respective folders:

- `cnn_cert/`: from https://github.com/IBM/CNN-Cert
- `convex_adversarial/`: from https://github.com/locuslab/convex_adversarial
- `crown_ibp/`: from https://github.com/huanzhang12/CROWN-IBP
- `eran/`: from https://github.com/eth-sri/eran
- `recurjac/`: from https://github.com/huanzhang12/RecurJac-and-CROWN



### Other Folders

To provide a uniform interface for them, we utilize the "adaptor design pattern", where we write a class to provide uniform methods for each approach in these tools. All classes are in folder `adaptor/`, and inherited from `adaptor.adaptor.Adaptor` class.

The `adaptor.adaptor.Adaptor` class should be initialized by dataset and model, and provides two main methods: `verify(self, input, label, norm_type, radius)` and `calc_radius(self, input, label, norm_type, upper=0.5, eps=1e-2)`.

The `verify()` method receives the input, true label, Lp norm type, and radius. It returns true or false. 

The `calc_radius()` method receives the input, true label, Lp norm type, the maximum possible radius, and the precision. It by default implements a binary-search based procedure which calls `verify()` multiples times to compute robustness radius.



The `basic/` folder includes our own reimplementation of a few verification approaches.

The `experiments/` folder contains our raw experiment data.

The `models/` folder contains the full definitions of the model structure used in the experiments. You can extend it to more models.

### Other Files

`constants.py` defines important global constants.

- If you want to improve the toolbox for more norm types, datasets, or verification approaches, remember to update them here.

`datasets.py` contains the dataset preparation scripts.

`model.py` indexes the models defined in `models/`, it contains a large dictionary, which maps the string indexes to concrete methods in `models/` which loads the models.

`main.py`: the toggle entrance for runing all the verification approaches.



## Copyrights

A significant amount of code in this project is embeded and adapted from existing open-sourced repositories. For those code, we keep all the source labels and author tags without modifications. We tried to list all the sources thoroughly above, but may still miss some. If you feel this tool violates your copyright, we apologize in advance and please contact us immediately.



For all other original parts, we allow free distribution of the code under the MIT license.



## Future Plans

We plan to provide an uniform interface for C++-based verification approaches, including Reluplex, Neurify, ReluVal, etc.

For recently popular randomized smoothing series approaches, we may provide a separate tool in the future.
