## bound_base.py
## 
## Common interface to all different bound computation methods (RecurJac, CROWN, fastlin and Fast-Lip)
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
from .bound_fastlin_fastlip import init_fastlin_bounds, fastlin_bound, fastlip_bound, fastlip_leaky_bound
from .bound_crown import crown_adaptive_bound, crown_general_bound, init_crown_bounds, compile_crown_bounds
from .bound_crown_quad import crown_quad_bound
from .bound_recurjac import recurjac_bound_wrapper, compile_recurjac_bounds


# use dictionary to save weights and bias
# use list to save "transposed" weights and bias
# e.g. for a 2 layer network with nodes 784 (input), 1024 (hidden), 10
# after transposed, shape of weights[0] = 1024*784, weights[1] = 10*1024 
def get_weights_list(model):
    
    weights = []
    bias = []
    
    U = model.U    
    for i, Ui in enumerate(U):
        # save hidden layer weights, layer by layer
        # middle layer weights: Ui
        [weight_Ui, bias_Ui] = Ui.get_weights()
        print("Hidden layer {} weight shape: {}".format(i, weight_Ui.shape))        
        weights.append(np.ascontiguousarray(np.transpose(weight_Ui)))
        bias.append(np.ascontiguousarray(np.transpose(bias_Ui)))
        print("Hidden layer {} bias shape: {}".format(i,bias_Ui.shape))

    # last layer weights: W
    [W, bias_W] = model.W.get_weights()
    weights.append(np.ascontiguousarray(np.transpose(W)))
    bias.append(np.ascontiguousarray(np.transpose(bias_W)))
    print("Last layer weight shape: {}".format(W.shape))
    print("Last layer bias shape: {}".format(bias_W.shape))

    for i, w in enumerate(weights):
        for p in [1,2,np.inf]:
            print("Layer {}, L_{} norm: {}".format(i, p, np.linalg.norm(w, p)))
    
    return weights, bias   

@jit(nopython=True)
def ReLU(vec):
    return np.maximum(vec, 0)

def compute_bounds_integral(weights, biases, pred_label, target_label, x0, predictions, numlayer, p, eps, steps, layerbndalg, jacbndalg, **kwargs):
    budget = None
    untargeted = kwargs.get("untargeted", False)
    for e in np.linspace(eps/steps, eps, steps):
        _, g_x0, max_grad_norm, _ = compute_bounds(weights, biases, pred_label, target_label, x0, predictions, numlayer, p, e, layerbndalg, jacbndalg, **kwargs)
        if budget is None:
            budget = g_x0
        new_budget = budget - max_grad_norm * (eps / steps)
        if untargeted:
            for j in range(weights[-1].shape[0]):
                if j < pred_label:
                    pass
                    # print("[L2] j = {}, validating eps={:.4f}, budget={:.4f}, new_budget={:.4f}, max_grad_norm={:.4f}".format(j, e, budget[j], new_budget[j], max_grad_norm[j]))
                elif j > pred_label:
                    pass
                    # print("[L2] j = {}, validating eps={:.4f}, budget={:.4f}, new_budget={:.4f}, max_grad_norm={:.4f}".format(j, e, budget[j-1], new_budget[j-1], max_grad_norm[j-1]))
        else:
            print("[L2] validating eps={:.4f}, budget={:.4f}, new_budget={:.4f}, max_grad_norm={:.4f}".format(e, budget[0], new_budget[0], max_grad_norm[0]))
        budget = new_budget
        if any(budget < 0):
            print("[L2][verification failure] min_perturbation = {:.4f}".format(e - eps/steps))
            return e - eps/steps
    print("[L2][verification success] eps = {:.4f}".format(e))
    return eps


def myprint(UB_Nk, LB_Nk):
    return
    np.set_printoptions(suppress=True)
    print('LB', LB_Nk)
    print('UB', UB_Nk)
    print('diff', ReLU(UB_Nk) - ReLU(LB_Nk))
    tight = LB_Nk * UB_Nk
    print('tight', tight)
    print(np.sum(np.minimum(tight, 0)))
    input()

def compute_bounds(weights, biases, pred_label, target_label, x0, predictions, numlayer, p, eps, layerbndalg, jacbndalg, **kwargs): 
    untargeted=kwargs.pop('untargeted', False)
    use_quad=kwargs.pop('use_quad', False)
    activation=kwargs.pop('activation', "relu")
    activation_param=kwargs.pop('activation_param', 0.3)
    lipsdir=kwargs.pop('lipsdir', -1)
    lipsshift=kwargs.pop('lipsshift', 1)
    bounded_input=kwargs.pop('bounded_input', False)
    assert len(kwargs) == 0, "unknow parameters " + str(kwargs)
    ### input example x0 
    # 784 by 1 (cifar: 3072 by 1)
    x0 = x0.flatten().astype(np.float32)
    # currently only supports p = "i"
    UB_N0 = x0 + eps
    LB_N0 = x0 - eps
    if bounded_input:
        UB_N0 = np.minimum(UB_N0, 1.0)
        LB_N0 = np.maximum(LB_N0, 0.0)
    
    # convert p into numba compatible form
    p_n = p
    q_n = int(1.0/ (1.0 - 1.0/p_n)) if p_n != 1 else np.inf
    
    # contains numlayer+1 arrays, each corresponding to a lower/upper bound 
    UBs = []
    LBs = []
    UBs.append(UB_N0)
    LBs.append(LB_N0)
    #save_bnd = {'UB_N0': UB_N0, 'LB_N0': LB_N0}
    neuron_states = []
    
       
    c = pred_label # c = 0~9
    j = target_label 

    # create diag matrices
    if layerbndalg == "crown-general":
        # construct linear upper and lower bounds
        bounds_ul = init_crown_bounds(weights)
        compile_crown_bounds(activation, activation_param)
    else:
        # construct gradient bounds
        assert activation == "relu", "activation function {} is only available for general mode".format(activation)
        diags = init_fastlin_bounds(weights)
    if jacbndalg == "recurjac":
        compile_recurjac_bounds(activation, activation_param)
    # print("Using {} activation".format(activation))

    
    ## weights and biases are already transposed
    if layerbndalg == "crown-general" or layerbndalg == "crown-adaptive" or layerbndalg == "fastlin" or \
            layerbndalg == "interval" or layerbndalg == "fastlin-interval" or layerbndalg == "crown-interval":
        # contains numlayer arrays, each corresponding to a pre-ReLU bound
        preReLU_UB = []
        preReLU_LB = []
        
        # for the first layer, we use a simple dual-norm based bound
        num = 0
        UB, LB = interval_bound(weights[num],biases[num],UBs[num],LBs[num], x0, eps, p_n)
        myprint(UB, LB)
        # save those pre-ReLU bounds
        preReLU_UB.append(UB)
        preReLU_LB.append(LB)
        # apply ReLU here manually (only used for computing neuron states)
        # for sigmoid family activations, this indicates the curvature
        UB = ReLU(UB)
        LB = ReLU(LB)
        neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
        # neurons never activated set to -1
        neuron_states[-1] -= UB == 0
        # neurons always activated set to +1
        neuron_states[-1] += LB > 0
        # print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
        #                     sum(neuron_states[-1] == +1), "neurons always activated")


        # we skip the last layer, which will be dealt later
        for num in range(1,numlayer-1):
            if layerbndalg == "interval":
                # all itermediate layers
                UB, LB = interval_bound(weights[num], biases[num], UB, LB, x0, eps, np.inf)
            if layerbndalg == "fastlin-interval":
                UB, LB = interval_bound(weights[num], biases[num], UB, LB, x0, eps, np.inf)
                # update diagonal matrix only, do not compute
                fastlin_bound(tuple(weights[:num+1]),tuple(biases[:num+1]),
                tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                tuple(neuron_states),
                num + 1,tuple(diags[:num+1]),
                x0,eps,p_n, skip = True)
            if layerbndalg == "fastlin":
                UB, LB = fastlin_bound(tuple(weights[:num+1]),tuple(biases[:num+1]),
                tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                tuple(neuron_states),
                num + 1,tuple(diags[:num+1]),
                x0,eps,p_n)
            if layerbndalg == "crown-interval":
                UB, LB = interval_bound(weights[num], biases[num], UB, LB, x0, eps, np.inf)
                crown_adaptive_bound(tuple(weights[:num+1]),tuple(biases[:num+1]),
                tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                tuple(neuron_states),
                num + 1,tuple(diags[:num+1]),
                x0,eps,p_n, skip = True)
            if layerbndalg == "crown-adaptive":
                UB, LB = crown_adaptive_bound(tuple(weights[:num+1]),tuple(biases[:num+1]),
                tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                tuple(neuron_states),
                num + 1,tuple(diags[:num+1]),
                x0,eps,p_n)
            if layerbndalg == "crown-general":
                UB, LB = crown_general_bound(tuple(weights[:num+1]),tuple(biases[:num+1]),
                tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                tuple(neuron_states),
                num + 1,tuple(bounds_ul[:num+1]),
                x0,eps,p_n)
            if num == 1 and use_quad:
                # apply quadratic bound
                UB_quad, LB_quad = crown_quad_bound(tuple(weights[:num+1]),tuple(biases[:num+1]),
                tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                tuple(neuron_states),
                num + 1,
                x0,eps,p_n)
                UB_prev = np.copy(UB)
                LB_prev = np.copy(LB)
                UB = np.minimum(UB, UB_quad)
                LB = np.maximum(LB, LB_quad)
                print("Quadratic bound improved {} of {} UBs".format(np.sum(UB_prev != UB), len(UB)))
                print("Quadratic bound improved {} of {} LBs".format(np.sum(LB_prev != LB), len(LB)))

            myprint(UB, LB)
            # last layer has no activation
            # save those pre-ReLU bounds
            preReLU_UB.append(UB)
            preReLU_LB.append(LB)
            # apply ReLU here manually (only used for computing neuron states)
            UB = ReLU(UB)
            LB = ReLU(LB)
            # Now UB and LB act just like before
            neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
            # neurons never activated set to -1
            neuron_states[-1] -= UB == 0
            # neurons always activated set to +1
            neuron_states[-1] += LB > 0
            # print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
            #                     sum(neuron_states[-1] == +1), "neurons always activated")

    else:
        raise(RuntimeError("unknown layerbndalg: {}".format(layerbndalg)))
                
    # form equavelent weight matrix to deal with the last layer
    num = numlayer - 1
    W = weights[num]
    bias = biases[num]
    if untargeted:
        ind = np.ones(len(W), bool)
        ind[c] = False
        W_last = W[c] - W[ind]
        b_last = bias[c] - bias[ind]
    else:
        if j == -1:
            # no targeted class, use class c only
            W_last = np.expand_dims(W[c], axis=0)
            b_last = np.expand_dims(bias[c], axis=0)
        else:
            W_last = np.expand_dims(W[c] - W[j], axis=0)
            b_last = np.expand_dims(bias[c] - bias[j], axis=0)
    if layerbndalg == "crown-general" or layerbndalg == "crown-adaptive" or layerbndalg == "fastlin" or layerbndalg == "interval" \
            or layerbndalg == "fastlin-interval" or layerbndalg == "crown-interval":
        if layerbndalg == "interval":
            UB, LB = interval_bound(W_last, b_last, UB, LB, x0, eps, np.inf)
        if layerbndalg == "fastlin" or layerbndalg == "fastlin-interval":
            # the last layer's weight has been replaced
            UB, LB = fastlin_bound(tuple(weights[:num]+[W_last]),tuple(biases[:num]+[b_last]),
            tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
            tuple(neuron_states),
            numlayer,tuple(diags),
            x0,eps,p_n)
        if layerbndalg == "crown-adaptive" or layerbndalg == "crown-interval":
            UB, LB = crown_adaptive_bound(tuple(weights[:num]+[W_last]),tuple(biases[:num]+[b_last]),
            tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
            tuple(neuron_states),
            numlayer,tuple(diags),
            x0,eps,p_n)
        if layerbndalg == "crown-general":
            UB, LB = crown_general_bound(tuple(weights[:num]+[W_last]),tuple(biases[:num]+[b_last]),
            tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
            tuple(neuron_states),
            numlayer,tuple(bounds_ul),
            x0,eps,p_n)
        # quadratic bound
        if use_quad:
            UB_quad, LB_quad = crown_quad_bound(tuple(weights[:num]+[W_last]),tuple(biases[:num]+[b_last]),
            tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
            tuple(neuron_states),
            numlayer,
            x0,eps,p_n)
            UB_prev = np.copy(UB)
            LB_prev = np.copy(LB)
            # print(UB_prev)
            # print(UB_quad)
            # print(LB_prev)
            # print(LB_quad)
            UB = min(UB_quad, UB)
            LB = max(LB_quad, LB)
            print("Quadratic bound improved {} of {} UBs".format(np.sum(UB_prev != UB), len(UB)))
            print("Quadratic bound improved {} of {} LBs".format(np.sum(LB_prev != LB), len(LB)))
        myprint(UB, LB)

    # Print bounds results
    # print("epsilon = {:.5f}".format(eps))
    # print("c = {}, {:.2f} < f_c < {:.2f}".format(c, LBs[numlayer][c], UBs[numlayer][c]))
    # print("j = {}, {:.2f} < f_j < {:.2f}".format(j, LBs[numlayer][j], UBs[numlayer][j])) 
    
    if untargeted:
        for j in range(W.shape[0]):
            if j < c:
                pass
                # print("    {:.4f} < f_c - f_{} < {:.4f}".format(LB[j], j, UB[j]))
            elif j > c:
                pass
                # print("    {:.4f} < f_c - f_{} < {:.4f}".format(LB[j-1], j, UB[j-1]))
        else:
            gap_gx = np.min(LB)
    else:
        print("    {:.4f} < f_c - f_j < {:.4f}".format(LB[0], UB[0]))
        gap_gx = LB[0]

    # Now "weights" are already transposed, so can pass weights directly to compute_max_grad_norm. 
    # Note however, if we transpose weights again, compute_max_grad_norm still works, but the result is different   
    # compute lipschitz bound
    if untargeted:
        g_x0 = []
        for j in range(W.shape[0]):
            if j < c:
                g_x0.append(predictions[c] - predictions[j])
            elif j > c:
                g_x0.append(predictions[c] - predictions[j])
    else:
        if j == -1:
            # no targeted class, use class c only
            g_x0 = [predictions[c]]
        else:
            g_x0 = [predictions[c] - predictions[j]]
    max_grad_norm = 0.0
    n_uns = float("nan")
    if jacbndalg == "fastlip":
        if activation == "relu":
            max_grad_norm, n_uns = fastlip_bound(tuple(weights[:num]+[W_last]), tuple(neuron_states), numlayer, q_n)
        elif activation == "leaky":
            max_grad_norm, n_uns = fastlip_leaky_bound(tuple(weights[:num]+[W_last]), tuple(neuron_states), numlayer, q_n, activation_param)
        if untargeted:
            for j in range(W.shape[0]):
                if j < c:
                    pass
                    # print("j = {}, g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(j, g_x0[j], max_grad_norm[j], g_x0[j] / max_grad_norm[j]))
                elif j > c:
                    pass
                    #print("j = {}".format(j))
                    #print("g_x0.shape = {}, max_grad_norm.shape = {}".format(len(g_x0),len(max_grad_norm)))
                    # print("j = {}, g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(j, g_x0[j-1], max_grad_norm[j-1], g_x0[j-1] / max_grad_norm[j-1]))
        else:
            print("g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(g_x0[0], max_grad_norm[0], g_x0[0] / max_grad_norm[0]))
    elif jacbndalg == "recurjac":
        max_grad_norm, n_uns = recurjac_bound_wrapper(tuple(weights[:num]+[W_last]), tuple(preReLU_UB), tuple(preReLU_LB), numlayer, norm = p_n, separated_bounds = untargeted, direction = lipsdir, shift = lipsshift)
        # for untargeted attack evaluation, we compute bounds for each output dimension separatedly
        if untargeted:
            for j in range(W.shape[0]):
                if j < c:
                    pass
                    # print("j = {}, g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(j, g_x0[j], max_grad_norm[j], g_x0[j] / max_grad_norm[j]))
                elif j > c:
                    pass
                    #print("j = {}".format(j))
                    #print("g_x0.shape = {}, max_grad_norm.shape = {}".format(len(g_x0),len(max_grad_norm)))
                    # print("j = {}, g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(j, g_x0[j-1], max_grad_norm[j-1], g_x0[j-1] / max_grad_norm[j-1]))
        # otherwise we consider the output as a vector and apply induced norm
        else:
            print("g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(g_x0[0], max_grad_norm[0], g_x0[0] / max_grad_norm[0]))

    sys.stdout.flush()
    sys.stderr.flush()
    return gap_gx, g_x0, max_grad_norm, n_uns


