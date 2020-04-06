## bound_fastlin_fastlip.py
## 
## Implementation of fastlin and Fast-Lip bounds
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import sys
from numba import jit, njit
import numpy as np
from .bound_interval import interval_bound


# bound a list of A matrix
def init_fastlin_bounds(Ws):
    nlayer = len(Ws)
    # preallocate all A matrices
    diags = [None] * nlayer
    # diags[0] is an identity matrix
    diags[0] = np.ones(Ws[0].shape[1], dtype=np.float32)
    for i in range(1,nlayer):
        diags[i] = np.empty(Ws[i].shape[1], dtype=np.float32)
    return diags


# matrix version of get_layer_bound_relax
@jit(nopython=True)
def fastlin_bound(Ws,bs,UBs,LBs,neuron_state,nlayer,diags,x0,eps,p_n,skip=False):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(diags)

    # step 1: create auxillary arrays; we have only nlayer-1 layers of activations
    # we only need to create for this new layer
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]

    # step 2: calculate all D matrices, there are nlayer such matrices
    # only need to create diags for this layer
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    alpha[idx_unsure] = UBs[nlayer-1][idx_unsure]/(UBs[nlayer-1][idx_unsure] - LBs[nlayer-1][idx_unsure])
    diags[nlayer-1][:] = alpha

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants = np.copy(bs[-1]) # the last bias

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants)
    LB_final = np.zeros_like(constants)
    if skip:
        return UB_final, LB_final
    # first A is W_{nlayer} D_{nlayer}
    A = Ws[nlayer-1] * diags[nlayer-1]
    for i in range(nlayer-1, 0, -1):
        # constants of previous layers
        constants += np.dot(A, bs[i-1])
        # unsure neurons of this layer
        idx_unsure = np.nonzero(neuron_state[i-1] == 0)[0]
        # create l array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        # bound the term A[i] * l_[i], for each element
        for j in range(A.shape[0]):
            l_ub[:] = 0.0
            l_lb[:] = 0.0
            # positive entries in j-th row, unsure neurons
            pos = np.nonzero(A[j][idx_unsure] > 0)[0]
            # negative entries in j-th row, unsure neurons
            neg = np.nonzero(A[j][idx_unsure] < 0)[0]
            # unsure neurons, corresponding to positive entries in A
            idx_unsure_pos = idx_unsure[pos]
            # unsure neurons, corresponding to negative entries in A
            idx_unsure_neg = idx_unsure[neg]
            # for upper bound, set the neurons with positive entries in A to upper bound
            # for upper bound, set the neurons with negative entries in A to lower bound, with l_ub[idx_unsure_neg] = 0
            l_ub[idx_unsure_pos] = LBs[i][idx_unsure_pos]
            # for lower bound, set the neurons with negative entries in A to upper bound
            # for lower bound, set the neurons with positive entries in A to lower bound, with l_lb[idx_unsure_pos] = 0
            l_lb[idx_unsure_neg] = LBs[i][idx_unsure_neg]
            # compute the relavent terms
            UB_final[j] -= np.dot(A[j], l_ub)
            LB_final[j] -= np.dot(A[j], l_lb)
        # compute A for next loop
        if i != 1:
            A = np.dot(A, Ws[i-1] * diags[i-1])
        else:
            A = np.dot(A, Ws[i-1]) # diags[0] is 1

    # now we have obtained A x + b_L <= f(x) <= A x + b_U
    # treat it as a one layer network and obtain bounds
    UB_first, LB_first = interval_bound(A, constants, UBs[0], LBs[0], x0, eps, p_n)
    UB_final += UB_first
    LB_final += LB_first

    return UB_final, LB_final

# W2 \in [c, M2], W1 \in [M2, M1]
# c, l, u \in [c, M1]
# r \in [c], k \in [M1], i \in [M2]
@jit(nopython=True)
def fastlip_2layer(W2, W1, neuron_state, norm = 1):
    # even if q_n != 1, then algorithm is the same. The difference is only at the output of fast_compute_max_grad_norm
    assert norm == 1
    # diag = 1 when neuron is active
    diag = np.maximum(neuron_state.astype(np.float32), 0)
    unsure_index = np.nonzero(neuron_state == 0)[0]
    # this is the constant part
    c = np.dot(diag * W2, W1)
    # this is the delta, and l <=0, u >= 0
    l = np.zeros((W2.shape[0], W1.shape[1]), dtype=W2.dtype)
    u = np.zeros_like(l)
    for r in range(W2.shape[0]):
        for k in range(W1.shape[1]):
            for i in unsure_index:
                prod = W2[r,i] * W1[i,k]
                if prod > 0:
                    u[r,k] += prod
                else:
                    l[r,k] += prod
    return c, l, u
    
# prev_c is the constant part; prev_l <=0, prev_u >= 0
# prev_c, prev_l, prev_u \in [c, M2], W1 \in [M2, M1]
# r \in [c], k \in [M1], i \in [M2]
@jit(nopython=True)
def fastlip_nplus1_layer(prev_c, prev_l, prev_u, W1, neuron_state, norm = 1):
    # c, l, u in shape(num_output_class, num_neurons_in_this_layer)
    c = np.zeros((prev_l.shape[0], W1.shape[1]), dtype = W1.dtype)
    l = np.zeros_like(c)
    u = np.zeros_like(c)
    # now deal with prev_l <= delta <= prev_u term
    # r is dimention for delta.shape[0]
    for r in range(prev_l.shape[0]):
        for k in range(W1.shape[1]):
            for i in range(W1.shape[0]):
                # unsure neurons
                if neuron_state[i] == 0:
                    if W1[i,k] > 0:
                        if W1[i,k] * (prev_c[r,i] + prev_u[r,i]) > 0:
                            u[r,k] += W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                        if W1[i,k] * (prev_c[r,i] + prev_l[r,i]) < 0:
                            l[r,k] += W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                    if W1[i,k] < 0:
                        if W1[i,k] * (prev_c[r,i] + prev_l[r,i]) > 0:
                            u[r,k] += W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                        if W1[i,k] * (prev_c[r,i] + prev_u[r,i]) < 0:
                            l[r,k] += W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                # active neurons
                if neuron_state[i] > 0:
                    # constant terms
                    c[r,k] += W1[i,k] * prev_c[r,i]
                    # upper/lower bounds terms
                    if W1[i,k] > 0:
                        u[r,k] += prev_u[r,i] * W1[i,k] 
                        l[r,k] += prev_l[r,i] * W1[i,k]
                    else:
                        u[r,k] += prev_l[r,i] * W1[i,k] 
                        l[r,k] += prev_u[r,i] * W1[i,k]
    return c, l, u

#@jit(nopython=True)
def fastlip_bound(weights, neuron_states, numlayer, norm):
    assert numlayer >= 2
    # merge the last layer weights according to c and j
    # W_vec = np.expand_dims(weights[-1][c] - weights[-1][j], axis=0)
    # const, l, u = fast_compute_max_grad_norm_2layer(W_vec, weights[-2], neuron_states[-1])
    const, l, u = fastlip_2layer(weights[-1], weights[-2], neuron_states[-1])
    # for layers other than the last two layers
    for i in list(range(numlayer - 2))[::-1]:
        const, l, u = fastlip_nplus1_layer(const, l, u, weights[i], neuron_states[i])
    # get the final upper and lower bound
    l += const
    u += const

    # count unsure elements
    p1 = l < 0
    p2 = u > 0
    uns = np.logical_and(p1, p2)
    n_uns = np.sum(uns)
    # print('Jacobian shape', l.shape)
    # print(u)
    # print('unsure elements:', n_uns)
    
    l = np.abs(l)
    u = np.abs(u)
    max_l_u = np.maximum(l, u)
    #print("max_l_u.shape = {}".format(max_l_u.shape))
    #print("max_l_u = {}".format(max_l_u))
    
    # the np.linalg.norm in numba does not support 'axis' parameter
    if norm == 1: 
        return np.sum(max_l_u, axis = 1), n_uns
    elif norm == 2: 
        return np.sqrt(np.sum(max_l_u**2, axis = 1)), n_uns
    elif norm == np.inf: # q_n = inf, return Li norm of max component
        # important: return type should be consistent with other returns 
        # For other 2 statements, it will return an array: [val], so we need to return an array.
        # numba doesn't support np.max and list, but support arrays
        max_ele = np.zeros((max_l_u.shape[0],))
        for i in range(max_l_u.shape[0]):
            for ii in range(max_l_u.shape[1]):
                if max_l_u[i][ii] > max_ele[i]:
                    max_ele[i] = max_l_u[i][ii]
        return max_ele, n_uns

@jit(nopython=True)
def fastlip_nplus1_layer_leaky(prev_c, prev_l, prev_u, W1, neuron_state, slope):
    # c, l, u in shape(num_output_class, num_neurons_in_this_layer)
    c = np.zeros((prev_l.shape[0], W1.shape[1]), dtype = W1.dtype)
    l = np.zeros_like(c)
    u = np.zeros_like(c)
    # now deal with prev_l <= delta <= prev_u term
    # r is dimention for delta.shape[0]
    for r in range(prev_l.shape[0]):
        for k in range(W1.shape[1]):
            for i in range(W1.shape[0]):
                # unsure neurons
                if neuron_state[i] == 0:
                    if W1[i,k] > 0:
                        if W1[i,k] * (prev_c[r,i] + prev_u[r,i]) > 0:
                            u[r,k] += W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                        else:
                            u[r,k] += slope * W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                        if W1[i,k] * (prev_c[r,i] + prev_l[r,i]) < 0:
                            l[r,k] += W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                        else:
                            l[r,k] += slope * W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                    if W1[i,k] < 0:
                        if W1[i,k] * (prev_c[r,i] + prev_l[r,i]) > 0:
                            u[r,k] += W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                        else:
                            u[r,k] += slope * W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                        if W1[i,k] * (prev_c[r,i] + prev_u[r,i]) < 0:
                            l[r,k] += W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                        else:
                            l[r,k] += slope * W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                # active neurons
                if neuron_state[i] > 0:
                    # constant terms
                    c[r,k] += W1[i,k] * prev_c[r,i]
                    # upper/lower bounds terms
                    if W1[i,k] > 0:
                        u[r,k] += prev_u[r,i] * W1[i,k] 
                        l[r,k] += prev_l[r,i] * W1[i,k]
                    else:
                        u[r,k] += prev_l[r,i] * W1[i,k] 
                        l[r,k] += prev_u[r,i] * W1[i,k]
                if neuron_state[i] < 0:
                    # constant terms
                    c[r,k] += slope * W1[i,k] * prev_c[r,i]
                    # upper/lower bounds terms
                    if W1[i,k] > 0:
                        u[r,k] += slope * prev_u[r,i] * W1[i,k] 
                        l[r,k] += slope * prev_l[r,i] * W1[i,k]
                    else:
                        u[r,k] += slope * prev_l[r,i] * W1[i,k] 
                        l[r,k] += slope * prev_u[r,i] * W1[i,k]
    return c, l, u

# W2 \in [c, M2], W1 \in [M2, M1]
# c, l, u \in [c, M1]
# r \in [c], k \in [M1], i \in [M2]
@jit(nopython=True)
def fastlip_2layer_leaky(W2, W1, neuron_state, slope):
    print("Computing Leaking ReLU max gradient norm")
    # diag = 1 when neuron is active, diag = slope when neuron is inactive, otherwise unsure (0)
    diag = neuron_state.astype(np.float32)
    # change -1 to +slope
    diag[diag < 0] *= - slope
    unsure_index = np.nonzero(neuron_state == 0)[0]
    # this is the constant part
    c = np.dot(diag * W2, W1)
    # this is the delta, and l <=0, u >= 0
    l = np.zeros((W2.shape[0], W1.shape[1]))
    u = np.zeros_like(l)
    for r in range(W2.shape[0]):
        for k in range(W1.shape[1]):
            for i in unsure_index:
                prod = W2[r,i] * W1[i,k]
                if prod > 0:
                    # make this neuron active in upper bound
                    u[r,k] += prod
                    # make this neuron inactive in lower bound
                    l[r,k] += slope * prod
                else:
                    # make this neuron active in lower bound
                    l[r,k] += prod
                    # make this neuron inactive in upper bound
                    u[r,k] += slope * prod
    # re-center
    m = (l + u) / 2
    c += m
    u -= m
    l -= m
    return c, l, u

def fastlip_leaky_bound(weights, neuron_states, numlayer, norm, slope):
    print("Computing Leaking ReLU max gradient norm")
    assert numlayer >= 2
    # merge the last layer weights according to c and j
    # W_vec = np.expand_dims(weights[-1][c] - weights[-1][j], axis=0)
    # const, l, u = fast_compute_max_grad_norm_2layer(W_vec, weights[-2], neuron_states[-1])
    const, l, u = fastlip_2layer_leaky(weights[-1], weights[-2], neuron_states[-1], slope)
    # for layers other than the last two layers
    for i in list(range(numlayer - 2))[::-1]:
        const, l, u = fastlip_nplus1_layer_leaky(const, l, u, weights[i], neuron_states[i], slope)
    # get the final upper and lower bound
    l += const
    u += const

    # count unsure elements
    p1 = l < 0
    p2 = u > 0
    uns = np.logical_and(p1, p2)
    n_uns = np.sum(uns)
    print('Jacobian shape', l.shape)
    # print(u)
    print('unsure elements:', n_uns)
    
    l = np.abs(l)
    u = np.abs(u)
    max_l_u = np.maximum(l, u)
    #print("max_l_u.shape = {}".format(max_l_u.shape))
    #print("max_l_u = {}".format(max_l_u))

    if norm == 1: # q_n = 1, return L1 norm of max component
        return np.sum(max_l_u, axis = 1), n_uns
    elif norm == 2: # q_n = 2, return L2 norm of max component
        return np.sqrt(np.sum(max_l_u**2, axis = 1)), n_uns
    elif norm == np.inf: # q_n = inf, return Li norm of max component
        # important: return type should be consistent with other returns 
        # For other 2 statements, it will return an array: [val], so we need to return an array.
        # numba doesn't support np.max and list, but support arrays
        max_ele = np.zeros((max_l_u.shape[0],))
        for i in range(max_l_u.shape[0]):
            for ii in range(max_l_u.shape[1]):
                if max_l_u[i][ii] > max_ele[i]:
                    max_ele[i] = max_l_u[i][ii]
        return max_ele, n_uns

