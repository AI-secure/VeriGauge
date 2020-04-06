#!/usr/bin/env python3
## main.py
## 
## Main command line interface for RecurJac, CRWON, Fast-Lip and Fast-Lip algorithms
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import os
import sys
import random
import argparse
import time
import tensorflow as tf
import numpy as np

from setup_mnist import MNIST
from setup_cifar import CIFAR
from mnist_cifar_models import NLayerModel, get_model_meta
from bound_base import get_weights_list
from utils import generate_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universal interface for RecurJac, CROWN, Fast-Lip and Fast-Lip algorithm experiments')
    parser.add_argument('--modelfile',
                required=True,
                default="",
                help='Path to a Keras model file. See train_nlayer.py for training a compatible model file.')
    parser.add_argument('--dataset', 
                default="auto",
                choices=["auto", "mnist", "cifar"],
                help='Dataset to be used. When set to "auto", it will automatically detect dataset based on the input dimension of model file.')
    parser.add_argument('--task',
                default="robustness",
                help='Define a task to run. This will call the "task_*.py" files under this directory. Currently supported tasks: robustness, landscape, lipschitz.')
    parser.add_argument('--numimage',
                default = 10,
                type = int,
                help='Number of images to run.')
    parser.add_argument('--startimage',
                default = 0,
                type = int,
                help='First image index in dataset.')
    parser.add_argument('--norm',
                default = "i",
                type = str,
                choices = ["i", "1", "2"],
                help='Perturbation norm: "i": Linf, "1": L1, "2": L2.')
    parser.add_argument('--layerbndalg',
                default = "crown-general",
                type = str,
                choices = ["crown-general", "crown-adaptive", "crown-interval", "fastlin", "interval", "fastlin-interval", "spectral"],
                help='Algorithm to compute layer-wise upper and lower bounds. "crown-general": CROWN for general activation functions, "crown-adaptive": CROWN for ReLU with adaptive upper and lower bounds, "fastlin": fastlin, "interval": Interval Bound Propagation, "spectral": spectral norm bounds (special, when use "spectral" bound we simply multiply each layer\'s operator norm).')
    parser.add_argument('--bounded-input',
                action='store_true',
                help='Use bounded input from 0 to 1 (be careful with data range!)')
    parser.add_argument('--jacbndalg',
                type = str,
                default = "disable",
                choices = ["disable", "recurjac", "fastlip"],
                help='Algorithm to compute Jacobian bounds. Used to compute (local) Lipschitz constant and robustness verification. When set to "disable", --layerbndalg will be used to compute robustness lower bound.')
    parser.add_argument('--lipsdir',
                type = int,
                default = -1,
                choices = [-1, +1],
                help='RecurJac bounding order, -1 backward, +1 forward. Usually set to -1.')
    parser.add_argument('--lipsshift',
                type = int,
                default = 1,
                choices = [0, +1],
                help='Shift RecurJac forward pass bounding by 1 layer (i.e., starting from layer 2 rather than layer 1; useful when the input layer has a large number of neurons). Usually set to 0.')
    parser.add_argument('--lipsteps',
                type = int,
                default = 15,
                help='Task specific. For the "lipschitz" task, this parameter specifies the number of eps values to evaluate local Lipschitz constants. For the "robustness" task, this parameter is the number of intervals for numerical integration; a larger value gives a better bound.')
    parser.add_argument('--eps',
                default = 0.005,
                type = float,
                help = 'Inital epsilon for "landscape" and "robustness" tasks.')
    parser.add_argument('--liplogstart',
                type = float,
                default = 0.0,
                help = 'Only used in "lipschitz" task. When LIPLOGSTART != LIPLOGSEND, we generate epsilon between LIPLOGSTART and LIPLOGEND using np.logspace with LIPSTEPS steps.')
    parser.add_argument('--liplogend',
                type = float,
                default = 0.0,
                help = 'See --liplogstart')
    parser.add_argument('--quad',
                action = "store_true",
                help='Use quadratic bound (for 2-layer ReLU network, CROWN-adaptive only).')
    parser.add_argument('--warmup',
                action = "store_true",
                help='Warm up before start timing. The first run will compile python code to native code, which takes a relatively long time, and should be excluded from timing.')
    parser.add_argument('--targettype',
                default="least",
                help='Target class label for robustness verification. Can be "least", "runnerup", "random" or "untargeted", or a number to specify class label.') 
    parser.add_argument('--steps',
                default = 15,
                type = int,
                help = 'Number of steps to do binary search.')
    parser.add_argument('--seed',
                default = 1228,
                type = int,
                help = 'Random seed.')

    args = parser.parse_args()
    # for all activations we can use general framework
    bounded_input = args.bounded_input
    if args.norm == "i":
        args.norm = np.inf
    else:
        args.norm = int(args.norm)

    targeted = True
    force_label = None
    if args.targettype == "least":
        target_type = 0b0100
    elif args.targettype == "runnerup":
        target_type = 0b0001
    elif args.targettype == "random":
        target_type = 0b0010
    elif args.targettype == "untargeted":
        target_type = 0b10000
        targeted = False
    elif args.targettype.isdecimal():
        # if target type is a number, force to use images from that class only
        target_type = None # don't care
        force_label = int(args.targettype)
    else:
        raise(ValueError("Unkonwn target type " + args.targettype))

    # try models/mnist_3layer_relu_1024
    modelfile = args.modelfile
    if not os.path.isfile(modelfile):
        raise(RuntimeError("cannot find model file"))
    # load Keras model
    weight_dims, activation, activation_param, input_dim = get_model_meta(modelfile)
    numlayer = len(weight_dims)
    if args.dataset == "auto":
        if input_dim[2] == 28 or input_dim[2] == "28":
            args.dataset = "mnist"
        elif input_dim[2] == 32:
            args.dataset = "cifar"
        else:
            raise(RuntimeError("Cannot determine dataset name for input with shape", input_dim))
    print('Loaded {} layer {} model with {} activation parameter = {}, input shape = {}'.format(len(weight_dims), args.dataset, activation, activation_param, input_dim))
    print('Model parameter dimensions:', weight_dims)
    assert args.layerbndalg == "crown-general" or args.layerbndalg == "spectral" or activation == "relu"
    # quadratic bound only works for ReLU
    assert ((not args.quad) or activation == "relu")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if args.dataset == "mnist":
            data = MNIST()
            model = NLayerModel(weight_dims[:-1], modelfile, activation=activation, activation_param=activation_param)
        elif args.dataset == "cifar":
            data = CIFAR()
            model = NLayerModel(weight_dims[:-1], modelfile, image_size=32, image_channel=3, activation=activation, activation_param=activation_param)
        else:
            raise(RuntimeError("unknown dataset: "+args.datset))
                
        print("Evaluating", modelfile)
        sys.stdout.flush()

        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

        # the weights and bias are saved in lists: weights and bias
        # weights[i-1] gives the ith layer of weight and so on
        weights, biases = get_weights_list(model)
        
        inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=data.test_labels.shape[0], total_images = args.numimage, targeted=targeted, random_and_least_likely = True, force_label = force_label, target_type = target_type, predictor=model.model.predict, start=args.startimage)
        # get the logit layer predictions
        preds = model.model.predict(inputs)

        task_input = locals()
        task_modudle = __import__("task_"+args.task)
        task = task_modudle.task(**task_input)

        # warmup
        if args.warmup:
            print("warming up...")
            task.warmup()
            sys.stdout.flush()

        print("Starting task!")
        sys.stdout.flush()
        sys.stderr.flush()
        total_time_start = time.time()

        for i in range(len(inputs)):
            single_time_start = time.time()
            task.run_single(i)
            sys.stdout.flush()
            sys.stderr.flush()
            print("[L1] time = {:.4f}".format(time.time() - single_time_start))
            
        print("[L0] total_time = {:.4f}".format(time.time() - total_time_start))
        task.summary()
        sys.stdout.flush()
        sys.stderr.flush()

