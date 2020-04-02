## task_landscape.py
## 
## Run RecurJac/FastLip bounds for exploring local optimization landscape
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
from bound_base import compute_bounds
from bound_spectral import spectral_bound

class task(object):
    def __init__(self, **kwargs):
        # add all arguments
        for k, v in kwargs.items():
            if not k.startswith("__"):
                exec('self.{} = kwargs["{}"]'.format(k, k))
        assert self.args.jacbndalg == "recurjac" or self.args.jacbndalg == "fastlip"
        assert self.args.layerbndalg == "crown-general" or self.args.layerbndalg == "crown-adaptive" or self.args.layerbndalg == "fastlin"
        assert self.targeted == True
        self.n_points = 0
        self.sum_max_eps = 0.0
        self.sum_lipschitz_max = 0.0
        print("starting stationary point discovery on {} images!".format(len(self.inputs)))

    def warmup(self, **kwargs):
        args = self.args
        compute_bounds(self.weights, self.biases, 0, -1, self.inputs[0], self.preds[0], self.numlayer,args.norm, 0.01, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)

    def _update_stats(self, current, lipschitz_const, n_uns):
        self.min_lipschitz = min(self.min_lipschitz, lipschitz_const)
        self.max_lipschitz = max(self.max_lipschitz, lipschitz_const)
        self.all_lipschitz[current] = lipschitz_const
        self.all_n_uns[current] = n_uns

    def run_single(self, i):
        args = self.args
        weights = self.weights
        biases = self.biases
        inputs = self.inputs
        preds = self.preds
        eps = args.eps

        self.n_points += 1
        predict_label = np.argmax(self.true_labels[i])
        target_label = -1
        start = time.time()
        self.all_lipschitz = defaultdict(float)
        self.all_n_uns = defaultdict(float)
        self.min_lipschitz = np.inf
        self.max_lipschitz = 0.0
        
        # binary search to find the largest eps that has at least one non-zero element
        def binary_search_cond(current):
            _, _, lipschitz_const, n_uns = compute_bounds(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, current, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
            return n_uns < weights[0].shape[1], n_uns

        if args.norm == 1:
            upper_limit = 100.0
        if args.norm == 2:
            upper_limit = 10.0
        if args.norm == np.inf:
            upper_limit = 1.0
        max_eps = binary_search(binary_search_cond, eps, upper_limit = upper_limit)
        self.sum_max_eps += max_eps

        # then do a linear scan with args.lipstep steps
        for current in np.linspace(0.0, max_eps, args.lipsteps + 1):
            _, _, lipschitz_const, n_uns = compute_bounds(weights, biases, predict_label, target_label, inputs[i], preds[i], self.numlayer, args.norm, current, args.layerbndalg, args.jacbndalg, untargeted = not self.targeted, use_quad = args.quad, activation = self.activation, activation_param = self.activation_param, lipsdir = args.lipsdir, lipsshift = args.lipsshift)
            self._update_stats(current, lipschitz_const[0], n_uns)

        s = []
        for current in np.linspace(0.0, max_eps, args.lipsteps + 1):
            s.append("lipschitz[{:.5f}] = {:.5f}".format(current, self.all_lipschitz[current]))
        print("[L1] " + ", ".join(s))
        s = []
        for current in np.linspace(0.0, max_eps, args.lipsteps + 1):
            s.append("uncertain[{:.5f}] = {:.5f}".format(current, self.all_n_uns[current]))
        print("[L1] " + ", ".join(s))
            
        self.sum_lipschitz_max += self.max_lipschitz

        # get the gradient at this data point
        gradients = self.model.get_gradient(inputs[i:i+1])
        obj_grad = gradients[predict_label]
        q = int(1.0/ (1.0 - 1.0/args.norm)) if args.norm != 1 else np.inf
        grad_norm = np.linalg.norm(obj_grad.flatten(), ord = q)
        predictions = self.model.model.predict(inputs[i:i+1])
        margin = predictions[0][predict_label]
        print("[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {}, info = {}, lipschitz_min = {:.5f}, lipschitz_max = {:.5f}, max_eps = {}, margin = {:.4f}, grad_norm = {:.4f}, time = {:.4f}".format(self.modelfile, i, self.true_ids[i], predict_label, target_label, self.img_info[i], self.min_lipschitz, self.max_lipschitz, max_eps, margin, grad_norm, time.time() - start))
        # check results
        assert(np.allclose(grad_norm, self.min_lipschitz))

    def summary(self, **kwargs):
        # compute and report global Lipschitz constant
        if self.force_label:
            _, lipschitz_const = spectral_bound(self.weights, self.biases, self.force_label, -1, self.inputs[0], self.preds[0], self.numlayer, self.activation, self.args.norm, not self.targeted)
            print("[L0] model = {}, numimage = {}, avg_max_eps = {:.5f}, avg_lipschitz_max = {:.4f}, opnorm_global_lipschitz = {:.4f}".format(self.modelfile, self.n_points, self.sum_max_eps / self.n_points, self.sum_lipschitz_max / self.n_points, lipschitz_const))
        else:
            print("[L0] model = {}, numimage = {}, avg_max_eps = {:.5f}, avg_lipschitz_max = {:.4f}".format(self.modelfile, self.n_points, self.sum_max_eps / self.n_points, self.sum_lipschitz_max / self.n_points))

