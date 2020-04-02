## task_lipschitz.py
## 
## Run RecurJac/FastLip bounds for computing (local) Lipschitz constant
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import time
import numpy as np
from collections import defaultdict
from utils import binary_search
from bound_base import compute_bounds, compute_bounds_integral
from bound_spectral import spectral_bound

class task(object):
    def __init__(self, **kwargs):
        # add all arguments
        for k, v in kwargs.items():
            if not k.startswith("__"):
                exec('self.{} = kwargs["{}"]'.format(k, k))
        assert self.args.jacbndalg == "recurjac" or (self.args.jacbndalg == "fastlip" and (self.activation == "relu" or self.activation == "leaky"))
        assert self.targeted == True
        self.n_points = 0
        self.sum_all_lipschitz = defaultdict(float)
        if self.args.liplogstart == self.args.liplogend:
            # use linear scale
            self.values = np.linspace(0.0, self.args.eps, self.args.lipsteps + 1)
        else:
            # use log scale
            self.values = [0] + list(np.logspace(self.args.liplogstart, self.args.liplogend, self.args.lipsteps))
        print("starting robustness verification on {} images!".format(len(self.inputs)))
        print("eps:", self.values)

    def warmup(self, **kwargs):
        args = self.args
        if args.layerbndalg == "spectral":
            spectral_bound(self.weights, self.biases, 0, -1, self.inputs[0], self.preds[0], self.numlayer, self.activation, args.norm, not self.targeted)
        else:
            compute_bounds(self.weights, self.biases, 0, -1, self.inputs[0], self.preds[0], self.numlayer,args.norm, 0.01, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)

    def _update_stats(self, current, lipschitz_const):
        self.min_lipschitz = min(self.min_lipschitz, lipschitz_const)
        self.max_lipschitz = max(self.max_lipschitz, lipschitz_const)
        self.all_lipschitz[current] = lipschitz_const
        self.sum_all_lipschitz[current] += lipschitz_const

    def run_single(self, i):
        args = self.args
        weights = self.weights
        biases = self.biases
        inputs = self.inputs
        preds = self.preds
        eps = args.eps

        self.n_points += 1
        predict_label = np.argmax(self.true_labels[i])
        # use this class only
        target_label = -1
        start = time.time()
        self.all_lipschitz = defaultdict(float)
        self.min_lipschitz = np.inf
        self.max_lipschitz = 0.0
        if args.layerbndalg == "spectral":
            _, lipschitz_const = spectral_bound(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, self.activation, args.norm, not self.targeted)
            self._update_stats(eps, lipschitz_const)
        # compute worst case bound
        # no need to pass in sess, model and data
        # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
        else:
            for current in self.values:
                print('[L2] current = {}'.format(current))
                _, _, lipschitz_const, _ = compute_bounds(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, current, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
                self._update_stats(current, lipschitz_const[0])

            s = []
            for current in self.values:
                s.append("lipschitz[{:.5f}] = {:.5f}".format(current, self.all_lipschitz[current]))
            print("[L1] " + ", ".join(s))
            
        # get the gradient at this data point
        gradients = self.model.get_gradient(inputs[i:i+1])
        obj_grad = gradients[predict_label]
        q = int(1.0/ (1.0 - 1.0/args.norm)) if args.norm != 1 else np.inf
        grad_norm = np.linalg.norm(obj_grad.flatten(), ord = q)
        predictions = self.model.model.predict(inputs[i:i+1])
        margin = predictions[0][predict_label]
        print("[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {}, info = {}, lipschitz_min = {:.5f}, lipschitz_max = {:.5f}, margin = {:.4f}, grad_norm = {:.4f}, time = {:.4f}".format(self.modelfile, i, self.true_ids[i], predict_label, target_label, self.img_info[i], self.min_lipschitz, self.max_lipschitz, margin, grad_norm, time.time() - start))
        # check results
        if args.layerbndalg != "spectral":
            assert(np.allclose(grad_norm, self.min_lipschitz))

    def summary(self, **kwargs):
        if self.args.layerbndalg == "spectral":
            lips_stats = "avg_global_lipschitz = {:.5f}".format(self.sum_all_lipschitz[self.args.eps] / self.n_points)
        else:
            s = []
            for current in self.values:
                s.append("avg_lipschitz[{:.5f}] = {:.5f}".format(current, self.sum_all_lipschitz[current] / self.n_points))
            lips_stats = ", ".join(s)
        if self.force_label:
            _, lipschitz_const = spectral_bound(self.weights, self.biases, self.force_label, -1, self.inputs[0], self.preds[0], self.numlayer, self.activation, self.args.norm, not self.targeted)
            print("[L0] model = {}, numimage = {}, {}, opnorm_global_lipschitz = {:.4f}".format(self.modelfile, self.n_points, lips_stats, lipschitz_const))
        else:
            print("[L0] model = {}, numimage = {}, {}".format(self.modelfile, self.n_points, lips_stats))

