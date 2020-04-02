## bound_crown.py
## 
## Implementation of CROWN-adaptive and CROWN-general bounds
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

from numba import jit, njit
import numpy as np
from .activation_functions import *
from .bound_interval import interval_bound

# for numba
linear_wrapper_constructed = False
leaky_wrapper = """
@njit
def get_leaky_relu_bounds_wrapper(UBs, LBs, neuron_states, bounds_ul, leaky_slope = {}): return get_leaky_relu_bounds(UBs, LBs, neuron_states, bounds_ul, leaky_slope)"""

sigmoid_family_ub_lb = """
@njit(cache=False)
def ub_pn(u, l):
    return general_ub_pn(u, l, {0}, {1})
@njit(cache=False)
def lb_pn(u, l):
    return general_lb_pn(u, l, {0}, {1})
@njit(cache=False)
def ub_p(u, l):
    return general_ub_p (u, l, {0}, {1})
@njit(cache=False)
def lb_p(u, l):
    return general_lb_p (u, l, {0}, {1})
@njit(cache=False)
def ub_n(u, l):
    return general_ub_n (u, l, {0}, {1})
@njit(cache=False)
def lb_n(u, l):
    return general_lb_n (u, l, {0}, {1})
"""

@jit(nopython=True,cache=False)
# def get_general_bounds(UBs, LBs, neuron_states, bounds_ul, ub_pn, lb_pn, ub_p, lb_p, ub_n, lb_n):
def get_general_bounds(UBs, LBs, neuron_states, bounds_ul):
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

# cannot unify the bounds calculation functions due to limitations of numba
@jit(nopython=True)
def get_relu_bounds(UBs, LBs, neuron_states, bounds_ul):
    ub_pn = relu_ub_pn
    lb_pn = relu_lb_pn
    ub_p  = relu_ub_p
    lb_p  = relu_lb_p
    ub_n  = relu_ub_n
    lb_n  = relu_lb_n
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

# cannot unify the bounds calculation functions due to limitations of numba
@jit(nopython=True)
def get_leaky_relu_bounds(UBs, LBs, neuron_states, bounds_ul, k):
    ub_pn = lambda u, l: leaky_relu_ub_pn(u, l, k = k)
    lb_pn = lambda u, l: leaky_relu_lb_pn(u, l, k = k)
    ub_p  = lambda u, l: leaky_relu_ub_p (u, l, k = k)
    lb_p  = lambda u, l: leaky_relu_lb_p (u, l, k = k)
    ub_n  = lambda u, l: leaky_relu_ub_n (u, l, k = k)
    lb_n  = lambda u, l: leaky_relu_lb_n (u, l, k = k)
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

def init_crown_bounds(Ws):
    nlayer = len(Ws)
    # preallocate all upper and lower bound slopes and intercepts
    bounds_ul = [None] * nlayer
    # first k is identity
    bounds_ul[0] = np.ones((4,Ws[0].shape[1]), dtype=np.float32)
    for i in range(1,nlayer):
        bounds_ul[i] = np.empty((4,Ws[i].shape[1]), dtype=np.float32)
    return bounds_ul

# we need to build this part of code dynamically due to the limit of numba
def compile_crown_bounds(activation, leaky_slope):
    global linear_wrapper_constructed
    if not linear_wrapper_constructed:
        if activation == "relu":
            exec("get_bounds = get_relu_bounds", globals())
        if activation == "leaky":
            code = compile(leaky_wrapper.format(leaky_slope), __file__, 'exec')
            exec(code, globals())
            exec("get_bounds = get_leaky_relu_bounds_wrapper", globals())
            print("Leaky ReLU Slope: {:.3f}".format(leaky_slope))
        elif activation == "tanh" or activation == "sigmoid" or activation == "arctan":
            exec(sigmoid_family_ub_lb.format("act_" + activation, "act_" + activation + "_d"), globals())
            exec("get_bounds = get_general_bounds", globals())
    linear_wrapper_constructed = True

# adaptive matrix version of get_layer_bound_relax
# get_bounds should be defined somewhere else (as a numba function)
@jit(nopython=True)
def crown_general_bound(Ws,bs,UBs,LBs,neuron_state,nlayer,bounds_ul,x0,eps,p_n):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(bounds_ul)
    assert p_n <= 2 or p_n == np.inf

    # step 2: compute slopes and intercepts for upper and lower bounds
    # only need to create upper/lower bounds' slope and intercept for this layer,
    # slopes and intercepts for previous layers have been stored
    # index: 0->slope for ub, 1->intercept for ub, 
    #        2->slope for lb, 3->intercept for lb
    get_bounds(UBs[nlayer-1], LBs[nlayer-1], neuron_state[nlayer - 2], bounds_ul[nlayer-1])

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants_ub = np.copy(bs[-1]) # the last bias
    constants_lb = np.copy(bs[-1]) # the last bias

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants_ub)
    LB_final = np.zeros_like(constants_lb)
    # first A is W_{nlayer} D_{nlayer}
    # A_UB = Ws[nlayer-1] * diags[nlayer-1]
    A_UB = np.copy(Ws[nlayer-1])
    # A_LB = Ws[nlayer-1] * diags[nlayer-1]
    A_LB = np.copy(Ws[nlayer-1])
    for i in range(nlayer-1, 0, -1):
        # create intercepts array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        diags_ub = np.empty_like(bounds_ul[i][0,:])
        diags_lb = np.empty_like(bounds_ul[i][0,:])
        upper_k = bounds_ul[i][0]
        upper_b = bounds_ul[i][1]
        lower_k = bounds_ul[i][2]
        lower_b = bounds_ul[i][3]
        # bound the term A[i] * l_[i], for each element
        for j in range(A_UB.shape[0]):
            # index for positive entries in A for upper bound
            idx_pos_ub = np.nonzero(A_UB[j] > 0)[0]
            # index for negative entries in A for upper bound
            idx_neg_ub = np.nonzero(A_UB[j] <= 0)[0]
            # index for positive entries in A for lower bound
            idx_pos_lb = np.nonzero(A_LB[j] > 0)[0]
            # index for negative entries in A for lower bound
            idx_neg_lb = np.nonzero(A_LB[j] <= 0)[0]
            # for upper bound, set the neurons with positive entries in A to upper bound
            diags_ub[idx_pos_ub]  = upper_k[idx_pos_ub]
            l_ub[idx_pos_ub] = upper_b[idx_pos_ub]
            # for upper bound, set the neurons with negative entries in A to lower bound
            diags_ub[idx_neg_ub] = lower_k[idx_neg_ub]
            l_ub[idx_neg_ub] = lower_b[idx_neg_ub]
            # for lower bound, set the neurons with negative entries in A to upper bound
            diags_lb[idx_neg_lb] = upper_k[idx_neg_lb]
            l_lb[idx_neg_lb] = upper_b[idx_neg_lb]
            # for lower bound, set the neurons with positve entries in A to lower bound
            diags_lb[idx_pos_lb] = lower_k[idx_pos_lb]
            l_lb[idx_pos_lb] = lower_b[idx_pos_lb]
            # compute the relavent terms
            UB_final[j] += np.dot(A_UB[j], l_ub)
            LB_final[j] += np.dot(A_LB[j], l_lb)
            # update the j-th row of A with diagonal matrice
            A_UB[j] = A_UB[j] * diags_ub
            # update A with diagonal matrice
            A_LB[j] = A_LB[j] * diags_lb
        # constants of previous layers
        constants_ub += np.dot(A_UB, bs[i-1])
        constants_lb += np.dot(A_LB, bs[i-1])
        # compute A for next loop
        # diags matrices is multiplied above
        A_UB = np.dot(A_UB, Ws[i-1])
        A_LB = np.dot(A_LB, Ws[i-1])
    # after the loop is done we get A0
    
    # now we have obtained A_L x + b_L <= f(x) <= A_U x + b_U
    # treat it as a one layer network and obtain bounds
    UB_first, _ = interval_bound(A_UB, constants_ub, UBs[0], LBs[0], x0, eps, p_n)
    _, LB_first = interval_bound(A_LB, constants_lb, UBs[0], LBs[0], x0, eps, p_n)
    UB_final += UB_first
    LB_final += LB_first
    
    return UB_final, LB_final

# adaptive matrix version of get_layer_bound_relax
@jit(nopython=True)
def crown_adaptive_bound(Ws,bs,UBs,LBs,neuron_state,nlayer,diags,x0,eps,p_n,skip = False):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(diags)

    # step 1: create auxillary arrays; we have only nlayer-1 layers of activations
    # we only need to create for this new layer
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]

    # step 2: calculate all D matrices, there are nlayer such matrices
    # only need to create diags for this layer
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    # prefill diags with u/(u-l)
    alpha[idx_unsure] = UBs[nlayer-1][idx_unsure]/(UBs[nlayer-1][idx_unsure] - LBs[nlayer-1][idx_unsure])
    diags[nlayer-1][:] = alpha

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants_ub = np.copy(bs[-1]) # the last bias
    constants_lb = np.copy(bs[-1]) # the last bias

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants_ub)
    LB_final = np.zeros_like(constants_lb)
    if skip:
        return UB_final, LB_final
    # first A is W_{nlayer} D_{nlayer}
    # A_UB = Ws[nlayer-1] * diags[nlayer-1]
    A_UB = np.copy(Ws[nlayer-1])
    # A_LB = Ws[nlayer-1] * diags[nlayer-1]
    A_LB = np.copy(Ws[nlayer-1])
    for i in range(nlayer-1, 0, -1):
        # unsure neurons of this layer
        idx_unsure = np.nonzero(neuron_state[i-1] == 0)[0]
        # create l array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        # bound the term A[i] * l_[i], for each element
        for j in range(A_UB.shape[0]):
            l_ub[:] = 0.0
            l_lb[:] = 0.0
            diags_ub = np.copy(diags[i])
            diags_lb = np.copy(diags[i])
            # positive entries in j-th row, unsure neurons
            pos_ub = np.nonzero(A_UB[j][idx_unsure] > 0)[0]
            pos_lb = np.nonzero(A_LB[j][idx_unsure] > 0)[0]
            # negative entries in j-th row, unsure neurons
            neg_ub = np.nonzero(A_UB[j][idx_unsure] < 0)[0]
            neg_lb = np.nonzero(A_LB[j][idx_unsure] < 0)[0]
            # unsure neurons, corresponding to positive entries in the j-th row of A
            idx_unsure_pos_ub = idx_unsure[pos_ub]
            idx_unsure_pos_lb = idx_unsure[pos_lb]
            # unsure neurons, corresponding to negative entries in the j-th row of A
            idx_unsure_neg_ub = idx_unsure[neg_ub]
            idx_unsure_neg_lb = idx_unsure[neg_lb]

            # for upper bound, set the neurons with positive entries in A to upper bound
            l_ub[idx_unsure_pos_ub] = LBs[i][idx_unsure_pos_ub]
            # for upper bound, set the neurons with negative entries in A to lower bound, depending on the magnitude of LBs[i][idx_unsure_neg] and UBs[i][idx_unsure_neg]
            mask = np.abs(LBs[i][idx_unsure_neg_ub]) > np.abs(UBs[i][idx_unsure_neg_ub])
            # for |LB| > |UB|, use y = 0 as the lower bound, adjust A
            diags_ub[idx_unsure_neg_ub[mask]] = 0.0
            # for |LB| < |UB|, use y = x as the lower bound, adjust A
            diags_ub[idx_unsure_neg_ub[np.logical_not(mask)]] = 1.0
            # update the j-th row of A with diagonal matrice
            A_UB[j] = A_UB[j] * diags_ub

            # for lower bound, set the neurons with negative entries in A to upper bound
            l_lb[idx_unsure_neg_lb] = LBs[i][idx_unsure_neg_lb]
            # for upper bound, set the neurons with positive entries in A to lower bound, depending on the magnitude of LBs[i][idx_unsure_pos] and UBs[i][idx_unsure_pos]
            mask = np.abs(LBs[i][idx_unsure_pos_lb]) > np.abs(UBs[i][idx_unsure_pos_lb])
            # for |LB| > |UB|, use y = 0 as the lower bound, adjust A
            diags_lb[idx_unsure_pos_lb[mask]] = 0.0
            # for |LB| > |UB|, use y = x as the lower bound, adjust A
            diags_lb[idx_unsure_pos_lb[np.logical_not(mask)]] = 1.0
            # update A with diagonal matrice
            A_LB[j] = A_LB[j] * diags_lb

            # compute the relavent terms
            UB_final[j] -= np.dot(A_UB[j], l_ub)
            LB_final[j] -= np.dot(A_LB[j], l_lb)
        # constants of previous layers
        constants_ub += np.dot(A_UB, bs[i-1])
        constants_lb += np.dot(A_LB, bs[i-1])
        # compute A for next loop
        A_UB = np.dot(A_UB, Ws[i-1])
        A_LB = np.dot(A_LB, Ws[i-1])
    # after the loop is done we get A0

    # now we have obtained A_L x + b_L f(x) <= A_U x + b_U
    # treat it as a one layer network and obtain bounds
    UB_first, _ = interval_bound(A_UB, constants_ub, UBs[0], LBs[0], x0, eps, p_n)
    _, LB_first = interval_bound(A_LB, constants_lb, UBs[0], LBs[0], x0, eps, p_n)
    UB_final += UB_first
    LB_final += LB_first

    return UB_final, LB_final

