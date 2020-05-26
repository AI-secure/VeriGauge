## bound_recurjac.py
## 
## Implementation of RecurJac bound
## 
## RecurJac is an efficient recursive algorithm for element-wise bounding Jacobian matrix of 
## neural networks with general activation functions
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import os
from numba import njit, prange, config
import numpy as np
import inspect
from .activation_functions import *

functions_constructed = False
grad_wrapper_constructed = False
grad_relu_wrapper = """
@njit(cache=False)
def get_grad_bounds_wrapper(lb, ub): 
    return relu_grad_bounds(lb, ub, alpha = {})
"""
grad_sigmoid_wrapper = """
@njit(cache=False)
def get_grad_bounds_wrapper(lb, ub):
    return sigmoid_grad_bounds(lb, ub, grad_f = {})
"""

config.NUMBA_NUM_THREADS = 20

# upper bound of sigmoid familiy's gradient, given a input range lb and ub
# returns: lower and upper bound of gradient in that region
@njit
def sigmoid_grad_bounds(lb, ub, grad_f):
    # allows a small tolerance
    assert lb <= ub + 1e-5
    if lb >= 0:
        # always positive
        return grad_f(ub), grad_f(lb)
    if ub <= 0:
        return grad_f(lb), grad_f(ub)
    # lb < 0, ub > 0
    return grad_f(max(-lb, ub)), grad_f(0)

# upper bound of ReLU familiy's gradient, given a input range lb and ub
# alpha is slope on the negative side. alpha > 0 for leaky-ReLU
# returns: lower and upper bound of gradient in that region
@njit
def relu_grad_bounds(lb, ub, alpha):
    # allows a small tolerance
    assert lb <= ub + 1e-5
    assert alpha >= 0
    if lb >= 0:
        return 1, 1
    if ub <= 0:
        return alpha, alpha
    else:
        return alpha, 1

# compute the bound of the last two layers
# W2 \in [c, M2], W1 \in [M2, M1]
# LB and UB are \in [M2]
# l, u \in [c, M1]
# r \in [c], k \in [M1], i \in [M2]
# output shape is [c, M1]
@njit(parallel=True)
def recurjac_bound_2layer(W2, W1, grad_UB, grad_LB):
    # this is the delta, and l <=0, u >= 0
    l = np.zeros((W2.shape[0], W1.shape[1]), dtype=W2.dtype)
    u = np.zeros_like(l)
    for r in prange(W2.shape[0]):
        for k in range(W1.shape[1]):
            # i is the index of neurons in the middle (M2)
            for i in range(W2.shape[1]):
                prod = W2[r,i] * W1[i,k]
                if prod > 0:
                    u[r,k] += grad_UB[i] * prod
                    l[r,k] += grad_LB[i] * prod
                else:
                    l[r,k] += grad_UB[i] * prod
                    u[r,k] += grad_LB[i] * prod
    return l, u

@njit(parallel=True)
def recurjac_bound_next_with_grad_bounds(prev_l, prev_u, W1, UB, LB):
    # l, u in shape(num_output_class, num_neurons_in_this_layer)
    l = np.zeros((prev_l.shape[0], W1.shape[1]), dtype = W1.dtype)
    u = np.zeros_like(l)
    # r is dimension for delta.shape[0]
    for r in prange(prev_l.shape[0]):
        for i in range(W1.shape[0]):
            grad_LB, grad_UB = get_grad_bounds(LB[i], UB[i])
            for k in range(W1.shape[1]):
                if W1[i,k] > 0:
                    if prev_u[r,i] > 0:
                        u[r,k] += grad_UB * W1[i,k] * prev_u[r,i]
                    else:
                        u[r,k] += grad_LB * W1[i,k] * prev_u[r,i]
                    if prev_l[r,i] < 0:
                        l[r,k] += grad_UB * W1[i,k] * prev_l[r,i]
                    else:
                        l[r,k] += grad_LB * W1[i,k] * prev_l[r,i]
                else:
                    if prev_l[r,i] > 0:
                        u[r,k] += grad_LB * W1[i,k] * prev_l[r,i]
                    else:
                        u[r,k] += grad_UB * W1[i,k] * prev_l[r,i]
                    if prev_u[r,i] < 0:
                        l[r,k] += grad_LB * W1[i,k] * prev_u[r,i]
                    else:
                        l[r,k] += grad_UB * W1[i,k] * prev_u[r,i]
    return l, u

# prev_l, prev_u are the bounds for previous layers
# prev_l, prev_u \in [c, M2], W1 \in [M2, M1]
# r \in [c], k \in [M1], i \in [M2]
# output in [c, M1]
@njit(parallel=True)
def recurjac_bound_next(prev_l, prev_u, W1, grad_UB, grad_LB):
    # l, u in shape(num_output_class, num_neurons_in_this_layer)
    l = np.zeros((prev_l.shape[0], W1.shape[1]), dtype = W1.dtype)
    u = np.zeros_like(l)
    # r is dimension for delta.shape[0]
    for r in prange(prev_l.shape[0]):
        for i in range(W1.shape[0]):
            for k in range(W1.shape[1]):
                if W1[i,k] > 0:
                    if prev_u[r,i] > 0:
                        u[r,k] += grad_UB[i] * W1[i,k] * prev_u[r,i]
                    else:
                        u[r,k] += grad_LB[i] * W1[i,k] * prev_u[r,i]
                    if prev_l[r,i] < 0:
                        l[r,k] += grad_UB[i] * W1[i,k] * prev_l[r,i]
                    else:
                        l[r,k] += grad_LB[i] * W1[i,k] * prev_l[r,i]
                else:
                    if prev_l[r,i] > 0:
                        u[r,k] += grad_LB[i] * W1[i,k] * prev_l[r,i]
                    else:
                        u[r,k] += grad_UB[i] * W1[i,k] * prev_l[r,i]
                    if prev_u[r,i] < 0:
                        l[r,k] += grad_LB[i] * W1[i,k] * prev_u[r,i]
                    else:
                        l[r,k] += grad_UB[i] * W1[i,k] * prev_u[r,i]
    return l, u

# improved recursive bound, handles 1 row and 1 column only (used for backtracking)
# w1 is a row vector
# select_column: only compute a specific column, this is the column for prev_l and prev_u
# neuron_masks: only handle selected neurons
# outputs are two scalars, l and u
@njit
def recurjac_bound_forward_improved_1x1(weights, w1, all_l, all_u, layer, gradUBs, gradLBs, select_column):
    l = 0
    u = 0
    grad_UB = gradUBs[layer]
    grad_LB = gradLBs[layer]
    # print("recursion layer", layer)
    # handle special case where we only have the last 2 layers to bound
    if layer == 0:
        for i in range(w1.shape[0]):
            prod = w1[i] * weights[0][i,select_column]
            if prod > 0:
                u += grad_UB[i] * prod ##!!UB
                l += grad_LB[i] * prod ##!!LB
            else:
                l += grad_UB[i] * prod ##!!LB
                u += grad_LB[i] * prod ##!!UB
        # print("layer is 0,", l.shape)
        return l, u
    # l, u in shape(num_output_class, num_neurons_in_this_layer)
    prev_l = all_l[layer]
    prev_u = all_u[layer]
    # r is dimension for delta.shape[0]
    r = select_column
    new_row_u = np.zeros_like(w1) ##!!UB
    new_row_l = np.zeros_like(w1) ##!!LB
    cu = 0
    cl = 0
    ce = 0
    # for each neuron
    for i in range(w1.shape[0]):
        # new row with activation function considered
        if w1[i] > 0:
            # this element is always negative, can propagate
            if prev_u[i,r] <= 0:
                cu += 1
                new_row_u[i] = w1[i] * grad_LB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            # this element is always positive, can propagate
            elif prev_l[i,r] >= 0:
                cl += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_LB[i] ##!!LB
            elif grad_UB[i] == grad_LB[i]:
                ce += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            else:
                # otherwise, we consider the worst case bound
                # where prev_u[i,r] > 0 AND prev_l[i,r] < 0
                u += grad_UB[i] * w1[i] * prev_u[i,r] ##!!UB
                l += grad_UB[i] * w1[i] * prev_l[i,r] ##!!LB
        else:
            # this element is always negative, can propagate
            if prev_u[i,r] <= 0:
                cu += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_LB[i] ##!!LB
            # this element is always positive, can propagate
            elif prev_l[i,r] >= 0:
                cl += 1
                new_row_u[i] = w1[i] * grad_LB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            # no difference between upper and lower bounds
            elif grad_UB[i] == grad_LB[i]:
                ce += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            else:
                # otherwise, we consider the worst case bound
                # where prev_u[i,r] > 0 AND prev_l[i,r] < 0
                u += grad_UB[i] * w1[i] * prev_l[i,r] ##!!UB
                l += grad_UB[i] * w1[i] * prev_u[i,r] ##!!LB
    if cu + cl + ce > 0:
        new_w_u = np.dot(np.expand_dims(new_row_u, axis=0), weights[layer]) ##!!UB
        new_w_l = np.dot(np.expand_dims(new_row_l, axis=0), weights[layer]) ##!!LB
        # now compute the bound for this 1-row vector only, with the previous layer
        _, u_kr = recurjac_bound_forward_improved_1x1_UB(weights, new_w_u[0], all_l, all_u, layer - 1, gradUBs, gradLBs, r) ##!!UB
        l_kr, _ = recurjac_bound_forward_improved_1x1_LB(weights, new_w_l[0], all_l, all_u, layer - 1, gradUBs, gradLBs, r) ##!!LB
        u += u_kr ##!!UB
        l += l_kr ##!!LB

    return l, u

# improved recursive bound, handles 1 row and 1 column only (used for backtracking)
# w1 is a column vector, but passed as a 1-d vector
# select_row: only compute a specific row, this is the row for prev_l and prev_u
# outputs are two scalars, l and u
@njit
def recurjac_bound_backward_improved_1x1(weights, w1, all_l, all_u, layer, gradUBs, gradLBs, select_row):
    l = 0
    u = 0
    grad_UB = gradUBs[layer]
    grad_LB = gradLBs[layer]
    # print(layer, grad_UB, grad_LB, w1.shape)
    # print("recursion layer", layer)
    # handle special case where we only have the first 2 layers to bound
    if layer == len(weights) - 2:
        for i in range(w1.shape[0]):
            prod = w1[i] * weights[-1][select_row,i]
            # print(w1[i], weights[-1][select_row,i], grad_UB[i])
            if prod > 0:
                u += grad_UB[i] * prod ##!!UB
                l += grad_LB[i] * prod ##!!LB
            else:
                l += grad_UB[i] * prod ##!!LB
                u += grad_LB[i] * prod ##!!UB
        # print("last layer", layer, l, u)
        return l, u
    # l, u in shape(num_output_class, num_neurons_in_this_layer)
    prev_l = all_l[layer+1]
    prev_u = all_u[layer+1]
    # weights of next layer
    next_W = weights[layer+1]
    # r is dimension for delta.shape[0]
    k = select_row
    new_row_u = np.zeros_like(w1) ##!!UB
    new_row_l = np.zeros_like(w1) ##!!LB
    cu = 0
    cl = 0
    ce = 0
    # for each neuron
    for i in range(prev_l.shape[1]):
        # new row with activation function considered
        if w1[i] > 0:
            # this element is always negative, can propagate
            if prev_u[k,i] <= 0:
                cu += 1
                new_row_u[i] = w1[i] * grad_LB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            # this element is always positive, can propagate
            elif prev_l[k,i] >= 0:
                cl += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_LB[i] ##!!LB
            elif grad_UB[i] == grad_LB[i]:
                ce += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            else:
                # otherwise, we consider the worst case bound
                # where prev_u[k,i] > 0 AND prev_l[k,i] < 0
                u += grad_UB[i] * w1[i] * prev_u[k,i] ##!!UB
                l += grad_UB[i] * w1[i] * prev_l[k,i] ##!!LB
        else:
            # this element is always negative, can propagate
            if prev_u[k,i] <= 0:
                cu += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_LB[i] ##!!LB
            # this element is always positive, can propagate
            elif prev_l[k,i] >= 0:
                cl += 1
                new_row_u[i] = w1[i] * grad_LB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            # no difference between upper and lower bounds
            elif grad_UB[i] == grad_LB[i]:
                ce += 1
                new_row_u[i] = w1[i] * grad_UB[i] ##!!UB
                new_row_l[i] = w1[i] * grad_UB[i] ##!!LB
            else:
                # otherwise, we consider the worst case bound
                # where prev_u[k,i] > 0 AND prev_l[k,i] < 0
                u += grad_UB[i] * w1[i] * prev_l[k,i] ##!!UB
                l += grad_UB[i] * w1[i] * prev_u[k,i] ##!!LB
    if cu + cl + ce > 0:
        # new_w_u and new_w_l is a column
        new_w_u = np.dot(next_W, np.expand_dims(new_row_u, axis=1)) ##!!UB
        new_w_l = np.dot(next_W, np.expand_dims(new_row_l, axis=1)) ##!!LB
        # print("layer", layer, l, u)
        # now compute the bound for this 1-row vector only, with the previous layer
        _, u_kr = recurjac_bound_backward_improved_1x1_UB(weights, new_w_u.reshape(-1), all_l, all_u, layer + 1, gradUBs, gradLBs, k) ##!!UB
        l_kr, _ = recurjac_bound_backward_improved_1x1_LB(weights, new_w_l.reshape(-1), all_l, all_u, layer + 1, gradUBs, gradLBs, k) ##!!LB
        u += u_kr ##!!UB
        l += l_kr ##!!LB
        # print("done layer", layer, l, u)

    return l, u

# improved recursive bound
# prev_l, prev_u are the bounds for previous layers
# prev_l, prev_u \in [c, M2], W1 \in [M2, M1]
# r \in [c], k \in [M1], i \in [M2]
# select_column: only compute a specific column
# output in [c, M1]
@njit(parallel=False)
def recurjac_bound_forward_improved(weights, W1, all_l, all_u, layer, gradUBs, gradLBs):
    # l, u in shape(num_output_class, num_neurons_in_this_layer)
    prev_l = all_l[layer]
    prev_u = all_u[layer]
    l = np.zeros((W1.shape[0], prev_l.shape[1]), dtype = W1.dtype)
    u = np.zeros_like(l)
    # weight of next layer is W1
    grad_UB = gradUBs[layer]
    grad_LB = gradLBs[layer]
    # weights of previous layer
    prev_W = weights[layer]
    # r is dimension for delta.shape[0]
    for k in prange(W1.shape[0]):
        for r in range(prev_l.shape[1]):
            new_row_u = np.zeros_like(W1[k,:])
            new_row_l = np.zeros_like(W1[k,:])
            cu = 0
            cl = 0
            ce = 0
            for i in range(W1.shape[1]):
                # new row with activation function considered
                if W1[k,i] > 0:
                    # this element is always negative, can propagate
                    if prev_u[i,r] <= 0:
                        cu += 1
                        new_row_u[i] = W1[k,i] * grad_LB[i]
                        new_row_l[i] = W1[k,i] * grad_UB[i]
                    # this element is always positive, can propagate
                    elif prev_l[i,r] >= 0:
                        cl += 1
                        new_row_u[i] = W1[k,i] * grad_UB[i]
                        new_row_l[i] = W1[k,i] * grad_LB[i]
                    # no difference between upper and lower bounds
                    elif grad_UB[i] == grad_LB[i]:
                        ce += 1
                        new_row_u[i] = W1[k,i] * grad_UB[i]
                        new_row_l[i] = W1[k,i] * grad_UB[i]
                    else:
                        # otherwise, we consider the worst case bound
                        # where prev_u[i,r] > 0 AND prev_l[i,r] < 0
                        u[k,r] += grad_UB[i] * W1[k,i] * prev_u[i,r]
                        l[k,r] += grad_UB[i] * W1[k,i] * prev_l[i,r]
                else:
                    # this element is always negative, can propagate
                    if prev_u[i,r] <= 0:
                        cu += 1
                        new_row_u[i] = W1[k,i] * grad_UB[i]
                        new_row_l[i] = W1[k,i] * grad_LB[i]
                    # this element is always positive, can propagate
                    elif prev_l[i,r] >= 0:
                        cl += 1
                        new_row_u[i] = W1[k,i] * grad_LB[i]
                        new_row_l[i] = W1[k,i] * grad_UB[i]
                    # no difference between upper and lower bounds
                    elif grad_UB[i] == grad_LB[i]:
                        ce += 1
                        new_row_u[i] = W1[k,i] * grad_UB[i]
                        new_row_l[i] = W1[k,i] * grad_UB[i]
                    else:
                        # otherwise, we consider the worst case bound
                        # where prev_u[i,r] > 0 AND prev_l[i,r] < 0
                        u[k,r] += grad_UB[i] * W1[k,i] * prev_l[i,r]
                        l[k,r] += grad_UB[i] * W1[k,i] * prev_u[i,r]
            # for each k and r, we need to form a new row vector, as the contents of the vector
            # depends on the sign of the k-th row and W and bounds of the r-th column of Y
            # after forming new_row_u and new_row_l, merge it with previous layer weights
            # print("layer", layer, "k", k, "r", r, "cu", cu, "cl", cl)
            if cu + cl + ce > 0:
                new_w_u = np.dot(np.expand_dims(new_row_u, axis=0), prev_W)
                new_w_l = np.dot(np.expand_dims(new_row_l, axis=0), prev_W)
                # now compute the bound for this 1-row vector only, with the previous layer
                _, u_kr = recurjac_bound_forward_improved_1x1_UB(weights, new_w_u[0], all_l, all_u, layer - 1, gradUBs, gradLBs, r)
                l_kr, _ = recurjac_bound_forward_improved_1x1_LB(weights, new_w_l[0], all_l, all_u, layer - 1, gradUBs, gradLBs, r)
                u[k,r] += u_kr
                l[k,r] += l_kr

    return l, u

# improved recursive bound
# prev_l, prev_u are the bounds for previous layers
# prev_l, prev_u \in [c, M2], W1 \in [M2, M1]
# r \in [c], k \in [M1], i \in [M2]
# select_column: only compute a specific column
# output in [c, M1]
@njit(parallel=False)
def recurjac_bound_backward_improved(weights, W1, all_l, all_u, layer, gradUBs, gradLBs):
    # l, u in shape(num_output_class, num_neurons_in_this_layer)
    prev_l = all_l[layer+1]
    prev_u = all_u[layer+1]
    l = np.zeros((prev_l.shape[0], W1.shape[1]), dtype = W1.dtype)
    u = np.zeros_like(l)
    # weight of next layer is W1
    grad_UB = gradUBs[layer]
    grad_LB = gradLBs[layer]
    # weights of next layer
    next_W = weights[layer+1]
    # r is dimension for delta.shape[0]
    for k in prange(prev_l.shape[0]):
        for r in range(W1.shape[1]):
            new_row_u = np.zeros_like(W1[:,r])
            new_row_l = np.zeros_like(W1[:,r])
            cu = 0
            cl = 0
            ce = 0
            for i in range(prev_l.shape[1]):
                # new row with activation function considered
                if W1[i,r] > 0:
                    # this element is always negative, can propagate
                    if prev_u[k,i] <= 0:
                        cu += 1
                        new_row_u[i] = W1[i,r] * grad_LB[i]
                        new_row_l[i] = W1[i,r] * grad_UB[i]
                    # this element is always positive, can propagate
                    elif prev_l[k,i] >= 0:
                        cl += 1
                        new_row_u[i] = W1[i,r] * grad_UB[i]
                        new_row_l[i] = W1[i,r] * grad_LB[i]
                    # no difference between upper and lower bounds
                    elif grad_UB[i] == grad_LB[i]:
                        ce += 1
                        new_row_u[i] = W1[i,r] * grad_UB[i]
                        new_row_l[i] = W1[i,r] * grad_UB[i]
                    else:
                        # otherwise, we consider the worst case bound
                        # where prev_u[k,i] > 0 AND prev_l[k,i] < 0
                        u[k,r] += grad_UB[i] * W1[i,r] * prev_u[k,i]
                        l[k,r] += grad_UB[i] * W1[i,r] * prev_l[k,i]
                else:
                    # this element is always negative, can propagate
                    if prev_u[k,i] <= 0:
                        cu += 1
                        new_row_u[i] = W1[i,r] * grad_UB[i]
                        new_row_l[i] = W1[i,r] * grad_LB[i]
                    # this element is always positive, can propagate
                    elif prev_l[k,i] >= 0:
                        cl += 1
                        new_row_u[i] = W1[i,r] * grad_LB[i]
                        new_row_l[i] = W1[i,r] * grad_UB[i]
                    # no difference between upper and lower bounds
                    elif grad_UB[i] == grad_LB[i]:
                        ce += 1
                        new_row_u[i] = W1[i,r] * grad_UB[i]
                        new_row_l[i] = W1[i,r] * grad_UB[i]
                    else:
                        # otherwise, we consider the worst case bound
                        # where prev_u[k,i] > 0 AND prev_l[k,i] < 0
                        u[k,r] += grad_UB[i] * W1[i,r] * prev_l[k,i]
                        l[k,r] += grad_UB[i] * W1[i,r] * prev_u[k,i]
            # for each k and r, we need to form a new row vector, as the contents of the vector
            # depends on the sign of the k-th row and W and bounds of the r-th column of Y
            # after forming new_row_u and new_row_l, merge it with previous layer weights
            # print("layer", layer, "k", k, "r", r, "cu", cu, "cl", cl)
            if cu + cl + ce > 0:
                # new_w_u and new_w_l is a column
                new_w_u = np.dot(next_W, np.expand_dims(new_row_u, axis=1))
                new_w_l = np.dot(next_W, np.expand_dims(new_row_l, axis=1))
                # now compute the bound for this 1-row vector only, with the previous layer
                _, u_kr = recurjac_bound_backward_improved_1x1_UB(weights, new_w_u.reshape(-1), all_l, all_u, layer + 1, gradUBs, gradLBs, k)
                l_kr, _ = recurjac_bound_backward_improved_1x1_LB(weights, new_w_l.reshape(-1), all_l, all_u, layer + 1, gradUBs, gradLBs, k)
                u[k,r] += u_kr
                l[k,r] += l_kr

    return l, u

# an optimized bound for the sum of absolute value in the first layer (computed at the end)
# last_l, last_u in shape [c,M1] 
# prev_l, prev_u in shape [c,M2] 
# last_W in shape [M2, M1]
# first_UB, first_LB in [M2]
def last_layer_abs_sum(weights, all_l, all_u, gradUBs, gradLBs, direction, shift):
    if direction == -1:
        last_l = all_l[0]
        last_u = all_u[0]
    else:
        last_l = all_l[-1]
        last_u = all_u[-1]
    first_W = weights[0]
    # add a new weights at the end
    s = np.empty(last_l.shape[0], dtype = last_l.dtype)
    # get the set where the Jacobian is always positive or negative
    for i in range(last_l.shape[0]):
        pos_idx = last_l[i] > 0
        neg_idx = last_u[i] < 0
        # print(sum(pos_idx), sum(neg_idx))
        uns_idx = np.logical_not(np.logical_or(pos_idx, neg_idx))
        # terms with unknown sign, use worst case bound
        unsure_term = np.sum(np.maximum(np.abs(last_l[i][uns_idx]), np.abs(last_u[i][uns_idx])))
        # now, use W_new to replace the first layer weight, and recompute the bound!
        if np.sum(uns_idx) != last_l.shape[1]:
            # merge the columns of first_W; after merging, its dimension becomes [M2, t], t < M1
            W_new = np.empty((first_W.shape[0], ), dtype = first_W.dtype)
            for r in range(first_W.shape[0]):
                W_new[r] = np.sum(first_W[r][pos_idx]) - np.sum(first_W[r][neg_idx])
            if direction == -1:
                _, tu = recurjac_bound_backward_improved_1x1_UB(weights, W_new, all_l, all_u, 0, gradUBs, gradLBs, i)
                s[i] = tu + unsure_term
            else:
                if shift == 0:
                    # replace the first layer weight
                    weights_new = list(weights)
                    # add a column vector to last
                    weights_new[0] = np.expand_dims(W_new, axis=1)
                    # we need to refill all_l and all_u as weights[0].shape[1] changed!
                    all_l_t = []
                    all_u_t = []
                    for j in range(len(weights_new)):
                        all_l_t.append(np.zeros(shape = (weights_new[j].shape[0], weights_new[0].shape[1]), dtype = weights_new[j].dtype))
                        all_u_t.append(np.zeros_like(all_l_t[-1]))
                    # for forward computation, we have to start from scratch
                    tl, tu = recurjac_bound(tuple(weights_new), tuple(gradUBs), tuple(gradLBs), tuple(all_l_t), tuple(all_u_t), direction)
                    s[i] = tu[0,0] + unsure_term
                elif shift == 1:
                    # shifted by 1; we computed layer 1 and 2 in backward mode, so we have Y(2) that can be directly used
                    tl, tu = recurjac_bound_next(all_l[-2], all_u[-2], np.expand_dims(W_new, axis=1), gradUBs[0], gradLBs[0])
                    s[i] = tu[0,0] + unsure_term

            # for debugging
            naive_bound = np.sum(np.maximum(np.abs(last_l[i]), np.abs(last_u[i])))
            saved = last_l.shape[1] - np.sum(uns_idx)
            # print("i =", i," naive_bound =", naive_bound, " improved_bound =", s[i], "saved =", saved)
        else:
            # print("i =", i, "saved = 0")
            s[i] = unsure_term
    return s

def construct_function(fn, suffix, mask):
    lines = inspect.getsource(fn)
    lines = lines.replace("def "+fn.__name__, "def "+fn.__name__+"_"+suffix)
    lines = lines.splitlines()
    for i in range(len(lines)):
        if lines[i].endswith("##!!"+mask):
            start = len(lines[i]) - len(lines[i].lstrip())
            lines[i] = lines[i][:start] + "# " + lines[i][start:]
    lines = "\n".join(lines)
    code = compile(lines, __file__, 'exec')
    exec(code, globals())

def compile_recurjac_bounds(activation, leaky_slope):
    global grad_wrapper_constructed
    if not grad_wrapper_constructed:
        # construct functions for gradient bounds dynamically
        if activation == "relu":
            code = compile(grad_relu_wrapper.format(0.0), __file__, 'exec')
            exec(code, globals())
        elif activation == "leaky":
            code = compile(grad_relu_wrapper.format(leaky_slope), __file__, 'exec')
            exec(code, globals())
        elif activation == "tanh" or activation == "sigmoid" or activation == "arctan":
            code = compile(grad_sigmoid_wrapper.format("act_"+activation+"_d"), __file__, 'exec')
            exec(code, globals())
        else:
            raise(ValueError("Unknown activation function"))
        grad_wrapper_constructed = True
    exec("get_grad_bounds = get_grad_bounds_wrapper", globals())

# for network with M layers, LBs dimension is M-1 (not including last layer)
def recurjac_bound_wrapper(weights, UBs, LBs, numlayer, norm, separated_bounds = False, direction = -1, shift = 1):
    global functions_constructed
    if not functions_constructed:
        construct_function(recurjac_bound_backward_improved_1x1, "UB", "LB")
        construct_function(recurjac_bound_backward_improved_1x1, "LB", "UB")
        construct_function(recurjac_bound_forward_improved_1x1, "UB", "LB")
        construct_function(recurjac_bound_forward_improved_1x1, "LB", "UB")
        functions_constructed = True
    assert shift >= 0
    assert shift <= 1
    if direction == +1:
        if shift == 1:
            print('forward bounding Lipschitz constant from layer {}, shape is'.format(shift+1), weights[shift].shape)
            # first, bound from layer shift, forward to the last layer
            # assume layer 2 is the input layer
            l, u, all_l, all_u, gradUBs, gradLBs = recurjac_bound_launcher(weights[shift:], UBs[shift:], LBs[shift:], numlayer - 1, norm, direction = +1)
            # then manually bound the first layer
            gradLBs = [np.empty_like(LBs[0])] + gradLBs
            gradUBs = [np.empty_like(UBs[0])] + gradUBs
            for r in prange(UBs[0].shape[0]):
                gradLBs[0][r], gradUBs[0][r] = get_grad_bounds(LBs[0][r], UBs[0][r])
            l, u = recurjac_bound_next_with_grad_bounds(l, u, weights[0], UBs[0], LBs[0])
            # update all_l and all_u
            # TODO: currently last l and u are placed at last element
            all_l.append(l)
            all_u.append(u)
        else:
            print('forward bounding Lipschitz constant, shape is', weights[0].shape)
            # only 3 layers, directly bound
            l, u, all_l, all_u, gradUBs, gradLBs = recurjac_bound_launcher(weights, UBs, LBs, numlayer, norm, direction = +1)
    elif direction == -1:
        # print('Backward bounding Lipschitz constant, shape is', weights[-1].shape)
        l, u, all_l, all_u, gradUBs, gradLBs = recurjac_bound_launcher(weights, UBs, LBs, numlayer, norm, direction = -1)

    # count unsure elements
    p1 = l < 0
    p2 = u > 0
    uns = np.logical_and(p1, p2)
    n_uns = np.sum(uns)
    # print('Jacobian shape', l.shape)
    # print(u)
    # print('unsure elements:', n_uns)
    
    return get_final_bound(weights, l, u, all_l, all_u, gradUBs, gradLBs, direction, shift, norm, separated_bounds), n_uns

@njit
def form_grad_bounds(UBs, LBs, gradUBs, gradLBs):
    # form gradient upper and lower bounds
    for i in range(len(UBs)):
        gradLB = gradLBs[i]
        gradUB = gradUBs[i]
        LB = LBs[i]
        UB = UBs[i]
        for r in range(UB.shape[0]):
            gradLB[r], gradUB[r] = get_grad_bounds(LB[r], UB[r])


def recurjac_bound_launcher(weights, UBs, LBs, numlayer, norm, direction):
    # pre-allocate arrays for lower and upper bounds
    all_l = []
    all_u = []
    if direction == -1:
        for i in range(numlayer):
            all_l.append(np.zeros(shape = (weights[numlayer-1].shape[0], weights[i].shape[1]), dtype = weights[i].dtype))
            all_u.append(np.zeros_like(all_l[-1]))
    elif direction == 1:
        for i in range(numlayer):
            all_l.append(np.zeros(shape = (weights[i].shape[0], weights[0].shape[1]), dtype = weights[i].dtype))
            all_u.append(np.zeros_like(all_l[-1]))
    # pre-allocate arrays for gradient lower and upper bounds
    gradUBs = []
    gradLBs = []
    for i in range(len(UBs)):
        gradUBs.append(np.empty_like(UBs[i]))
        gradLBs.append(np.empty_like(LBs[i]))
    # form gradient upper and lower bounds
    form_grad_bounds(tuple(UBs), tuple(LBs), tuple(gradUBs), tuple(gradLBs))
    # compute the bound
    l, u = recurjac_bound(weights, tuple(gradUBs), tuple(gradLBs), tuple(all_l), tuple(all_u), direction)
    return l, u, all_l, all_u, gradUBs, gradLBs
    

# if the output dimension is c, when separated is True, output is c individual bounds
# when separated is False, output is a scalar and it is the Lipschitz constant for the norm of the output
# direct -1: from last to first layers; direction 1: from first to last layers
@njit(parallel=True)
def recurjac_bound(weights, gradUBs, gradLBs, all_l, all_u, direction):
    # print('This is RecurJac Jacobian bound!')
    numlayer = len(weights)
    assert numlayer >= 2
    assert direction == +1 or direction == -1

    # gradXB[i] represent the bounds between i+1 and i+2 layer
    # all_l[i] represent Y related to layer i+1
    if direction == -1:
        # form L and U from the last layer, they are just the last layer weight
        all_l[-1][:] = weights[-1]
        all_u[-1][:] = weights[-1]
        l, u = recurjac_bound_2layer(weights[-1], weights[-2], gradUBs[-1], gradLBs[-1])
        # L and U for numlayer-1
        all_l[-2][:] = l
        all_u[-2][:] = u
        # print("layer", numlayer - 2, "shape", l.shape)
        # for layers other than the last two layers
        # layer 2, 1, 0, etc
        for i in list(range(numlayer - 2))[::-1]:
            # get bound for layer i+2 (Y) and i+1 (W)
            l, u = recurjac_bound_backward_improved(weights, weights[i], all_l, all_u, i, gradUBs, gradLBs)
            all_l[i][:] = l
            all_u[i][:] = u
            # print("layer", i, "shape", l.shape)
        # get the final upper and lower bound
        l = all_l[0]
        u = all_u[0]
    elif direction == 1:
        # precreated L and U for each layer
        # form L and U from the first layer
        all_l[0][:] = weights[0]
        all_u[0][:] = weights[0]
        # layer 1 and 2
        l, u = recurjac_bound_2layer(weights[1], weights[0], gradUBs[0], gradLBs[0])
        all_l[1][:] = l
        all_u[1][:] = u
        # we need to transpose l and U to reuse recurjac_bound_next()
        # print("layer", 0, "shape", all_l[1].shape)
        # for layers other than the last two layers
        # layer 2 and 3, 3 and 4, etc
        for i in list(range(1, numlayer - 1)):
            # note that W is transposed
            # get bound for layer i+1 (Y) and i+2 (W)
            l, u = recurjac_bound_forward_improved(weights, weights[i+1], all_l, all_u, i, gradUBs, gradLBs)
            all_l[i + 1][:] = l
            all_u[i + 1][:] = u
            # print("layer", i, "shape", all_l[i+1].shape)
        # get the final upper and lower bound
        l = all_l[-1]
        u = all_u[-1]
    return l, u

def get_final_bound(weights, l, u, all_l, all_u, gradUBs, gradLBs, direction, shift, norm, separated_bounds = False):
    M = np.maximum(np.abs(l), np.abs(u))

    if separated_bounds:
        # We look each row individually
        if norm == 1: 
            # Linf norm (max) of each row
            ret = np.empty((M.shape[0],), dtype = M.dtype)
            for i in range(M.shape[0]):
                ret[i] = np.max(M[i])
            return ret
        elif norm == 2:
            # L2 norm of each row
            return np.sqrt(np.sum(M**2, axis = 1))
        elif norm == np.inf: # q_n = inf, return Li norm of max component
            # L1 norm of each row
            # return np.sum(M, axis = 1)
            return last_layer_abs_sum(weights, all_l, all_u, gradUBs, gradLBs, direction, shift)
    else:
        # look for the matrix induced norm
        # we need to unify return type for numba, so we return an array
        ret = np.empty((1,), dtype = M.dtype)
        if norm == 1: 
            # L1 induced norm of matrix M
            ret[0] = np.linalg.norm(M, norm)
        elif norm == 2:
            # L2 induced norm of matrix M
            ret[0] = np.linalg.norm(M, norm)
        elif norm == np.inf:
            # Linf induced norm of M
            # Linf induced norm is the max column abs sum
            # ret[0] = np.linalg.norm(M, norm)
            ret[0] = np.max(last_layer_abs_sum(weights, all_l, all_u, gradUBs, gradLBs, direction, shift))
        return ret

