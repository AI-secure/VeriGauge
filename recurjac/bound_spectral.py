## bound_spectral.py
## 
## Implementation of operator norm based global Lipschitz constant
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import numpy as np
import time

norm_prev = 1.0

# spectral norm bound for global lipschitz constant
def spectral_bound(weights, biases, pred_label, target_label, x0, predictions, numlayer, activation = "relu", p=np.inf, untargeted = False):
    global norm_prev
    c = pred_label # c = 0~9
    j = target_label 
    print("norm is", p)
    norm = 1.0
    if norm_prev == 1.0:
        # compute hidden layer spectral norms
        for l in range(numlayer - 1):
            print(weights[l].shape)
            layer_norm = np.linalg.norm(weights[l], ord = p)
            print("{} norm of layer {} is {}".format(p, l, layer_norm))
            norm *= layer_norm
            if activation == "relu" or activation == "leaky":
                pass
            elif activation == "tanh":
                pass
            elif activation == "arctan":
                pass
            elif activation == "sigmoid":
                norm *= 0.25
            else:
                raise(ValueError("Unknown activation function " + activation))
        norm_prev = norm
        print("Layer n-1 norm {}".format(norm))
    else:
        norm = norm_prev
        print("using cached norm {}".format(norm))
    # form the last layer weights
    num = numlayer - 1
    W = weights[num]
    bias = biases[num]
    if untargeted:
        ind = np.ones(len(W), bool)
        ind[c] = False
        W_lasts = W[c] - W[ind]
        print("last layer shape", W_lasts.shape)
        # norms of last layers
        last_norms = []
        for W_last in W_lasts:
            W_last = np.expand_dims(W_last, axis=0)
            last_norms.append(np.linalg.norm(W_last, ord = p))
        # g_x0 of last layers
        g_x0s = []
        for jj in range(W.shape[0]):
            if jj != c:
                g_x0s.append(predictions[c] - predictions[jj])
        # compute bounds
        bnds = []
        for last_norm, g_x0 in zip(last_norms, g_x0s):
            bnds.append(g_x0 / (last_norm * norm))
        print("Untargeted norm, g_x0 and bnd:")
        print(["{0:8.5g}".format(i) for i in last_norms])
        print(["{0:8.5g}".format(i) for i in g_x0s])
        print(["{0:8.5g}".format(i) for i in bnds])
        # find the minimum bound
        return min(bnds), last_norms
    else:
        if j == -1:
            # no targeted class, use class c only
            W_last = np.expand_dims(W[c], axis=0)
        else:
            W_last = np.expand_dims(W[c] - W[j], axis=0)
        """
        print('++++++++++++++++++++++++++')
        print(W_last.shape)
        print(W_last)
        print(np.linalg.norm(W_last, ord = p))
        print(np.linalg.norm(W[c] - W[j], ord = p))
        print(np.linalg.norm(W[c] - W[j], ord = 1))
        print('--------------------------')
        """
        last_norm = np.linalg.norm(W_last, ord = p)
        if j == -1:
            print("{} norm of last layer class {} is {}".format(p, c, last_norm))
        else:
            print("{} norm of last layer class {} - {} is {}".format(p, c, j, last_norm))
        total_norm = norm * last_norm
        g_x0 = predictions[c] - predictions[j]
        print("total_norm = {}, g_x0 = {} - ({}) = {}".format(total_norm, predictions[c], predictions[j], g_x0))
        return g_x0 / total_norm, total_norm
    
