#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_bound_ours.py

core functions for fastlin and Fast-Lip bounds

Copyright (C) 2018, Lily Weng  <twweng@mit.edu>
                    Huan Zhang <ecezhang@ucdavis.edu>
                    Honge Chen <chenhg@mit.edu>
"""


from numba import jit
import numpy as np
#from get_bounds_others import get_layer_bound_LP
from scipy.sparse import csr_matrix

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
        weights.append(csr_matrix(np.ascontiguousarray(np.transpose(weight_Ui))))
        bias.append(np.ascontiguousarray(np.transpose(bias_Ui)))
        print("Hidden layer {} bias shape: {}".format(i,bias_Ui.shape))

    # last layer weights: W
    [W, bias_W] = model.W.get_weights()    
    weights.append(csr_matrix(np.ascontiguousarray(np.transpose(W))))
    bias.append(np.ascontiguousarray(np.transpose(bias_W)))
    print("Last layer weight shape: {}".format(W.shape))
    print("Last layer bias shape: {}".format(bias_W.shape))
    
    return weights, bias   

@jit(nopython=True)
def ReLU(vec):
    
    return np.maximum(vec, 0)

@jit
def get_layer_bound(W_Nk,b_Nk,UB_prev,LB_prev,is_last,x0,eps,p_n):
    W_NkA = W_Nk.A
    gamma = np.empty(W_NkA.shape, dtype=W_NkA.dtype)
    # gamma = np.transpose(gamma)
    eta = np.empty_like(gamma)
    
    UB_Nk = np.empty_like(b_Nk)
    LB_Nk = np.empty_like(b_Nk)
    
    UB_new = np.empty_like(b_Nk)
    LB_new = np.empty_like(b_Nk)

    #print("W_Nk shape")
    #print(W_Nk.shape)
    
    # I reordered the indices for faster sequential access, so gamma and eta are now transposed
    for ii in range(W_NkA.shape[0]):
        for jj in range(W_NkA.shape[1]):
            if W_NkA[ii,jj] > 0:
                gamma[ii,jj] = UB_prev[jj]
                eta[ii,jj] = LB_prev[jj]
            else:
                gamma[ii,jj] = LB_prev[jj]
                eta[ii,jj] = UB_prev[jj]
              
        UB_Nk[ii] = np.dot(W_NkA[ii],gamma[ii])+b_Nk[ii]
        LB_Nk[ii] = np.dot(W_NkA[ii],eta[ii])+b_Nk[ii]
        #print('UB_Nk[{}] = {}'.format(ii,UB_Nk[ii]))
        #print('LB_Nk[{}] = {}'.format(ii,LB_Nk[ii]))
    
    Ax0 = W_Nk.dot(x0)
    dualnorm_Aj = np.zeros(W_NkA.shape[0])
    for j in range(W_Nk.shape[0]):
        if p_n == 105: # p == "i", q = 1
            dualnorm_Aj = np.sum(np.abs(W_NkA[j]))
        elif p_n == 1: # p = 1, q = i
            dualnorm_Aj = np.max(np.abs(W_NkA[j]))
        elif p_n == 2: # p = 2, q = 2
            dualnorm_Aj = np.linalg.norm(W_NkA[j])    
        UB_new[j] = Ax0[j]+eps*dualnorm_Aj+b_Nk[j]
        LB_new[j] = Ax0[j]-eps*dualnorm_Aj+b_Nk[j]

    #if p_n == 105: # p == "i", q = 1
    #    dualnorm_Aj = abs(W_Nk).sum(1).A
    #elif p_n == 1: # p = 1, q = i
    #    dualnorm_Aj = abs(W_Nk).max(1).A
    #elif p_n == 2: # p = 2, q = 2
    #    dualnorm_Aj = np.sqrt(W_Nk.power(2).sum(1)).A
    
    
    #for j in range(W_Nk.shape[0]):
    #    UB_new[j] = Ax0[j]+eps*dualnorm_Aj[j]+b_Nk[j]
    #    LB_new[j] = Ax0[j]-eps*dualnorm_Aj[j]+b_Nk[j]
    is_old = False

    if is_last: # the last layer has no ReLU
        if is_old:
            return UB_Nk, LB_Nk
        else:
            return UB_new, LB_new
    else:# middle layers
        return ReLU(UB_Nk), ReLU(LB_Nk)

# bound a list of A matrix
def init_layer_bound_relax_matrix_huan(Ws):
    nlayer = len(Ws)
    # preallocate all A matrices
    diags = [None] * nlayer
    # diags[0] will be skipped
    diags[0] = np.ones(1, dtype=np.float32)
    for i in range(1,nlayer):
        diags[i] = np.empty(Ws[i].shape[1], dtype=np.float32)
    return diags

# matrix version of get_layer_bound_relax
@jit
def get_layer_bound_relax_matrix_huan_optimized(Ws,bs,UBs,LBs,neuron_state,nlayer,diags,x0,eps,p_n):
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
    # first A is W_{nlayer} D_{nlayer}
    A = csr_matrix(Ws[nlayer-1].multiply(diags[nlayer-1]))
    for i in range(nlayer-1, 0, -1):
        # constants of previous layers
        constants += A.dot(bs[i-1])
        # unsure neurons of this layer
        idx_unsure = np.nonzero(neuron_state[i-1] == 0)[0]
        # create l array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        # bound the term A[i] * l_[i], for each element
        AA = A.A
        for j in range(A.shape[0]):
            l_ub[:] = 0.0
            l_lb[:] = 0.0
            pos = np.nonzero(AA[j][idx_unsure] > 0)[0]
            neg = np.nonzero(AA[j][idx_unsure] < 0)[0]
            idx_unsure_pos = idx_unsure[pos]
            idx_unsure_neg = idx_unsure[neg]
            l_ub[idx_unsure_pos] = LBs[i][idx_unsure_pos]
            l_lb[idx_unsure_neg] = LBs[i][idx_unsure_neg]
            UB_final[j] -= np.dot(AA[j], l_ub)
            LB_final[j] -= np.dot(AA[j], l_lb)
        # compute A for next loop
        if i != 1:
            A = csr_matrix(A.dot(Ws[i-1].multiply(diags[i-1])))
        else:
            A = csr_matrix(A.dot(Ws[i-1])) # diags[0] is 1
    # after the loop is done we get A0
    UB_final += constants
    LB_final += constants

    # step 6: bounding A0 * x
    x_UB = np.empty_like(UBs[0])
    x_LB = np.empty_like(LBs[0])


    Ax0 = A.dot(x0)
    A = A.A
    if p_n == 105: # means p == "i":     
        #dualnorm_Aj = abs(A).sum(1).A
        for j in range(A.shape[0]):        
            dualnorm_Aj = np.sum(np.abs(A[j])) # L1 norm of A[j]
            UB_final[j] += (Ax0[j]+eps*dualnorm_Aj)
            LB_final[j] += (Ax0[j]-eps*dualnorm_Aj)

    elif p_n == 1: # means p == "1"
        #dualnorm_Aj = abs(A).max(1).A
        for j in range(A.shape[0]):        
            dualnorm_Aj = np.max(np.abs(A[j])) # Linf norm of A[j]
            UB_final[j] += (Ax0[j]+eps*dualnorm_Aj)
            LB_final[j] += (Ax0[j]-eps*dualnorm_Aj)

    elif p_n == 2: # means p == "2"
        #dualnorm_Aj = np.sqrt(A.power(2).sum(1)).A
        for j in range(A.shape[0]):        
            dualnorm_Aj = np.linalg.norm(A[j]) # L2 norm of A[j]
            UB_final[j] += (Ax0[j]+eps*dualnorm_Aj)
            LB_final[j] += (Ax0[j]-eps*dualnorm_Aj)
    
    return UB_final, LB_final



"""
Main computing function 1
"""
def compute_worst_bound(weights, biases, pred_label, target_label, x0, predictions, numlayer, p="i", eps = 0.005, method="ours", lipsbnd="disable", is_LP=False, is_LPFULL=False,untargeted = False,dual=False):
    ### input example x0 
    # 784 by 1 (cifar: 3072 by 1)
    x0 = x0.flatten().astype(np.float32)
    # currently only supports p = "i"
    UB_N0 = x0 + eps
    LB_N0 = x0 - eps
    
    # convert p into numba compatible form
    if p == "i":
        p_n = ord('i') # 105
        q_n = 1 # the grad_norm 
    elif p == "1":
        p_n = 1
        q_n = ord('i') # 105
    elif p == "2":
        p_n = 2
        q_n = 2
    else:
        print("currently doesn't support p = {}, only support p = i,1,2".format(p))
    
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
    diags = init_layer_bound_relax_matrix_huan(weights)
    
    if method == "ours" or is_LPFULL:
        # contains numlayer arrays, each corresponding to a pre-ReLU bound
        preReLU_UB = []
        preReLU_LB = []
        
        for num in range(numlayer):
            
            # first time compute the bound of 1st layer
            if num == 0: # get the raw bound
                UB, LB = get_layer_bound(weights[num],biases[num],UBs[num],LBs[num],True,x0,eps,p_n)
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)
                # apply ReLU here manually
                UB = ReLU(UB)
                LB = ReLU(LB)

                neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,", 
                                    sum(neuron_states[-1] == +1), "neurons always activated")                               
                UBs.append(UB)
                LBs.append(LB)              
                
            # we skip the last layer, which will be dealt later
            elif num != numlayer - 1:
                
                if is_LPFULL:
                    UB, LB, _ = get_layer_bound_LP(weights[:num+1],biases[:num+1],[UBs[0]]+preReLU_UB,
                    [LBs[0]]+preReLU_LB,x0,eps,p,neuron_states,num+1,c,j,True,False,dual)
                else:
                    # UB, LB = get_layer_bound_relax_matrix_huan(weights[:num+1],biases[:num+1],
                    UB, LB = get_layer_bound_relax_matrix_huan_optimized(tuple(weights[:num+1]),tuple(biases[:num+1]),
                    tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                    tuple(neuron_states),
                    num + 1,tuple(diags[:num+1]),
                    x0,eps,p_n)
                # last layer has no activation
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)
                # apply ReLU here manually
                UB = ReLU(UB)
                LB = ReLU(LB)
                # Now UB and LB act just like before
                neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,", 
                                    sum(neuron_states[-1] == +1), "neurons always activated")
                UBs.append(UB)
                LBs.append(LB)

    else:
        raise(RuntimeError("unknown method number: {}".format(method)))
                
        
    num = numlayer - 1
    W = weights[num]
    bias = biases[num]
    if untargeted:
        ind = np.ones(len(W), bool)
        ind[c] = False
        W_last = W[c] - W[ind]
        b_last = bias[c] - bias[ind]
    else:
        W_last = np.expand_dims(W.A[c] - W.A[j], axis=0)
        W_last = csr_matrix(W_last)
        b_last = np.expand_dims(bias[c] - bias[j], axis=0)
    
    if method == "ours":
        # UB, LB = get_layer_bound_relax_matrix_huan(weights[:num]+[W_last],biases[:num]+[b_last],
        #            [UBs[0]]+preReLU_UB,[LBs[0]]+preReLU_LB,
        #            neuron_states,
        #            True, numlayer)
        UB, LB = get_layer_bound_relax_matrix_huan_optimized(tuple(weights[:num]+[W_last]),tuple(biases[:num]+[b_last]),
        tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
        tuple(neuron_states),
        numlayer,tuple(diags),
        x0,eps,p_n)

    # Print bounds results
    print("epsilon = {:.5f}".format(eps))
    # print("c = {}, {:.2f} < f_c < {:.2f}".format(c, LBs[numlayer][c], UBs[numlayer][c]))
    # print("j = {}, {:.2f} < f_j < {:.2f}".format(j, LBs[numlayer][j], UBs[numlayer][j])) 
    
    # after all bounds has been computed, we run a LP to find the last layer bounds
    if is_LP or is_LPFULL:
        if untargeted:
            LP_UB, LP_LB, LP_LBs = get_layer_bound_LP(weights,biases,[UBs[0]]+preReLU_UB,
            [LBs[0]]+preReLU_LB, x0, eps, p, neuron_states,numlayer,c,j,False,True,dual)
            LP_UBs = np.empty_like(LP_LBs)
            LP_UBs[:] = np.inf
        else:
            LP_UB, LP_LB, LP_bnd_gx0 = get_layer_bound_LP(weights,biases,[UBs[0]]+preReLU_UB,
            [LBs[0]]+preReLU_LB, x0, eps, p, neuron_states,numlayer,c,j,False,False,dual)
        # print("c = {}, {:.2f} < f_c < {:.2f}".format(c, LP_LB[c], LP_UB[c]))
        # print("j = {}, {:.2f} < f_j < {:.2f}".format(j, LP_LB[j], LP_UB[j])) 
        
    if untargeted:
        for j in range(W.shape[0]):
            if j < c:
                print("    {:.2f} < f_c - f_{} < {:.2f}".format(LB[j], j, UB[j]))
                if is_LP:
                    print("LP  {:.2f} < f_c - f_{} < {:.2f}".format(LP_LBs[j], j, LP_UBs[j]))
            elif j > c:
                print("    {:.2f} < f_c - f_{} < {:.2f}".format(LB[j-1], j, UB[j-1]))
                if is_LP:
                    print("LP  {:.2f} < f_c - f_{} < {:.2f}".format(LP_LBs[j-1], j, LP_UBs[j-1]))
        if is_LP or is_LPFULL:
            gap_gx = np.min(LP_LBs)
        else:
            gap_gx = np.min(LB)
    else:
        if is_LP or is_LPFULL:
            print("    {:.2f} < f_c - f_j ".format(LP_bnd_gx0))
            gap_gx = LP_bnd_gx0
        else:
            print("    {:.2f} < f_c - f_j < {:.2f}".format(LB[0], UB[0]))
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
        g_x0 = [predictions[c] - predictions[j]]
    max_grad_norm = 0.0

    return gap_gx, g_x0, max_grad_norm


