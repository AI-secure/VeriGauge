#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_bound_ours.py

core functions for Fast-Lin and Fast-Lip bounds

Copyright (C) 2018, Lily Weng  <twweng@mit.edu>
                    Huan Zhang <ecezhang@ucdavis.edu>
                    Honge Chen <chenhg@mit.edu>
"""

import numpy as np
import time
import gurobipy as grb
from gurobipy import GRB

norm_prev = 1.0

"""
Spectral norm bound
"""
def spectral_bound(weights, biases, pred_label, target_label, x0, predictions, numlayer, p="i", untargeted = False):
    global norm_prev
    c = pred_label # c = 0~9
    j = target_label 
    if p == "i":
        p = np.inf
    else:
        p = int(p)
    norm = 1.0
    if norm_prev == 1.0:
        # compute hidden layer spectral norms
        for l in range(numlayer - 1):
            print(weights[l].shape)
            layer_norm = np.linalg.norm(weights[l], ord = p)
            print("{} norm of layer {} is {}".format(p, l, layer_norm))
            norm *= layer_norm
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
        return min(bnds)
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
        print("{} norm of last layer is {}".format(p, last_norm))
        total_norm = norm * last_norm
        g_x0 = predictions[c] - predictions[j]
        print("total_norm = {}, g_x0 = {} - ({}) = {}".format(total_norm, predictions[c], predictions[j], g_x0))
        return g_x0 / total_norm
    
"""
LP bound
"""
def get_layer_bound_LP(Ws,bs,UBs,LBs,x0,eps,p,neuron_states,nlayer,pred_label,target_label,compute_full_bounds=False,untargeted=False, dual=False):
    # storing upper and lower bounds for last layer
    UB = np.empty_like(bs[-1])
    LB = np.empty_like(bs[-1])
    # neuron_state is an array: neurons never activated set to -1, neurons always activated set to +1, indefinite set to 0    
    # indices
    alphas = []
    # for n layer network, we have n-1 layers of relu
    for i in range(nlayer-1):
        idx_unsure = (neuron_states[i] == 0).nonzero()[0]
        # neuron_state is an integer array for efficiency reasons. We should convert it to float
        alpha = neuron_states[i].astype(np.float32)
        alpha[idx_unsure] = UBs[i+1][idx_unsure]/(UBs[i+1][idx_unsure]-LBs[i+1][idx_unsure])
        alphas.append(alpha)
    
    start = time.time()
    m = grb.Model("LP")
    m.setParam("outputflag",0)
    # disable parallel Gurobi solver, using 1 thread only
    if dual:
        m.setParam("Method",1)
    else:
        m.setParam("Method",-1)
    m.setParam("Threads", 1) # only 1 thread
    # z and zh are list of lists, each list for one layer of variables
    # z starts from 1, matching Zico's notation
    z = []
    z.append(None)
    # z hat starts from 2
    zh = []
    zh.append(None)
    zh.append(None)
    
    if p == "2" or p == "1":
        # ztrans (transformation of z1 only for lp norm), starts from 1 matching z
        ztrans = []
        ztrans.append(None)
    
    ## LP codes: 
    
    # we start our label from 1 to nlayer+1 (the last one is the final objective layer)
    # valid range for z: 1 to nlayer (z_1 is just input, z_{nlayer} is the last relu layer output)
    # valid range for z_hat: 2 to nlayer+1 (there is no z_hat_1 as it is the input, z_{nlayer+1} is final output)
    for i in range(1,nlayer+2):
        if i == 1: # first layer
            # first layer, only z exists, no z hat
            zzs = []
            zzts = []
            # UBs[0] is for input x. Create a variable for each input
            # and set its lower and upper bounds
            for j in range(1,len(UBs[0])+1):
                zij = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=LBs[0][j-1], ub=UBs[0][j-1], name="z_"+str(i)+"_"+str(j))
                zzs.append(zij)
                if p == "2" or p == "1":                
                    # transformation variable at z1 only
                    if p == "2":
                        ztij = m.addVar(vtype=grb.GRB.CONTINUOUS, name="zt_"+str(i)+"_"+str(j))
                    elif p == "1":
                        ztij = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name="zt_"+str(i)+"_"+str(j))
                    zzts.append(ztij)  
            z.append(zzs)
            if p == "2" or p == "1":
                ztrans.append(zzts)
        elif i< nlayer+1:
            # middle layer, has both z and z hat
            zzs = []
            zzhs = []
            for j in range(1,len(UBs[i-1])+1):
                zij = m.addVar(vtype=grb.GRB.CONTINUOUS, name="z_"+str(i)+"_"+str(j))
                zzs.append(zij)
                
                zhij = m.addVar(vtype=grb.GRB.CONTINUOUS,lb=-np.inf,name="zh_"+str(i)+"_"+str(j))
                zzhs.append(zhij)
            z.append(zzs)
            zh.append(zzhs)
        else: # last layer, i == nlayer + 1
            # only has z hat, length is the same as the output
            # there is no relu, so no z
            zzhs = []
            for j in range(1,len(bs[-1])+1):
                zhij = m.addVar(vtype=grb.GRB.CONTINUOUS,lb=-np.inf,name="zh_"+str(i)+"_"+str(j))
                zzhs.append(zhij)
            zh.append(zzhs)
    
    m.update()

    # Adding weights constraints for all layers
    for i in range(1,nlayer+1):
        W = Ws[i-1] # weights of layer i
        for j in range(W.shape[0]):
            """
            sum_term = bs[i-1][j]
            for s in range(W.shape[1]):
                # z start from 1
                sum_term += z[i][s]*W[j,s]
            """
            sum_term = grb.LinExpr(W[j], z[i]) + bs[i-1][j]
            # this is the output of layer i, and let z_hat_{i+1} equal to it
            # z_hat_{nlayer+1} is the final output (logits)
            m.addConstr(sum_term == zh[i+1][j], "weights==_"+str(i)+"_"+str(j))
            # m.addConstr(sum_term <= zh[i+1][j], "weights<=_"+str(i)+"_"+str(j))
            # m.addConstr(sum_term >= zh[i+1][j], "weights>=_"+str(i)+"_"+str(j))
    
    # nlayer network only has nlayer - 1 activations
    for i in range(1, nlayer):
        # UBs[0] is the bounds for input x, so start from 1
        for j in range(len(UBs[i])):
            # neuron_states starts from 0
            if neuron_states[i-1][j] == 1:
                m.addConstr(z[i+1][j] == zh[i+1][j], "LPposr==_"+str(j))
                # m.addConstr(z[i+1][j] <= zh[i+1][j], "LPpos<=_"+str(j))
                # m.addConstr(z[i+1][j] >= zh[i+1][j], "LPpos>=_"+str(j))
            elif neuron_states[i-1][j] == -1:
                m.addConstr(z[i+1][j] == 0, "LPneg==_"+str(j))
                # m.addConstr(z[i+1][j] <= 0, "LPneg<=_"+str(j))
                # m.addConstr(z[i+1][j] >= 0, "LPneg>=_"+str(j))
            elif neuron_states[i-1][j] == 0:
                # m.addConstr(z[i+1][j] >= 0, "LPunsure>=0_"+str(j))
                m.addConstr(z[i+1][j] >= zh[i+1][j], "LPunsure>=_"+str(j))
                m.addConstr(z[i+1][j] <= alphas[i-1][j]*(zh[i+1][j]-LBs[i][j]), "LPunsure<=_"+str(j))
            else:
                raise(RuntimeError("unknown neuron_state: "+neuron_states[i])) 

    
#    #finally, add constraints for z[1], the input -> For p == "i", this is already added in the input variable range zij
#    for i in range(len(UBs[0])):
#         m.addConstr(z[1][i] <= UBs[0][i], "inputs+_"+str(i))
#         m.addConstr(z[1][i] >= LBs[0][i], "inputs-_"+str(i))         

    if p == "2": 
        #finally, add constraints for z[1] and ztrans[1], the input
        for i in range(len(UBs[0])):
            m.addConstr(ztrans[1][i] == z[1][i] - x0[i], "INPUTtrans_"+str(i))
        # quadratic constraints
        m.addConstr(grb.quicksum(ztrans[1][i]*ztrans[1][i] for i in range(len(UBs[0]))) <= eps*eps, "INPUT L2 norm QCP")
    elif p == "1":
         #finally, add constraints for z[1] and ztrans[1], the input
        temp = []
        for i in range(len(UBs[0])):
            tempi = m.addVar(vtype=grb.GRB.CONTINUOUS)
            temp.append(tempi)
                    
        for i in range(len(UBs[0])):
            # absolute constraints: seem that option1 and 2a, 2c are the right answer (compared to p = 2 result) 
            # option 1
            #m.addConstr(ztrans[1][i] >= z[1][i] - x0[i], "INPUTtransPOS_"+str(i))
            #m.addConstr(ztrans[1][i] >= -z[1][i] + x0[i], "INPUTtransNEG_"+str(i))
            
            # option 2a: same answer as option 1
            # note if we write , the result is different
            #zzz = m.addVar(vtype=grb.GRB.CONTINUOUS)
            #m.addConstr(zzz == z[1][i]-x0[i])
            #m.addConstr(ztrans[1][i] == grb.abs_(zzz), "INPUTtransABS_"+str(i))
            
            # option 2b: gives different sol as 2a and 2c, guess it's because abs_() has to take a variable,
            # and that's why 2a and 2c use additional variable zzz or temp
            # but now it gives Attribute error on "gurobipy.LinExpr", so can't use this anymore
            #m.addConstr(ztrans[1][i] == grb.abs_(z[1][i]-x0[i]), "INPUTtransABS_"+str(i))
            
            # option 2c: same answer as 2a
            m.addConstr(temp[i] == z[1][i]-x0[i])
            m.addConstr(ztrans[1][i] == grb.abs_(temp[i]), "INPUTtransABS_"+str(i))    
            
            # option 3: same answer as 2b
            #m.addConstr(ztrans[1][i] <= z[1][i] - x0[i], "INPUTtransPOS_"+str(i))
            #m.addConstr(ztrans[1][i] >= -z[1][i] + x0[i], "INPUTtransNEG_"+str(i))


        # L1 constraints
        m.addConstr(grb.quicksum(ztrans[1][i] for i in range(len(UBs[0]))) <= eps, "INPUT L1 norm")
            
    

    # another way to write quadratic constraints
    ###expr = grb.QuadExpr()
    ###expr.addTerms(np.ones(len(UBs[0])), z[1], z[1])
    ###m.addConstr(expr <= eps*eps)

    m.update()

    print("[L2][LP solver initialized] time_lp_init = {:.4f}".format(time.time() - start))
    # for middle layers, need to compute full bounds
    if compute_full_bounds:
        # compute upper bounds        
        # z_hat_{nlayer+1} is the logits (final output, or inputs for layer nlayer+1)
        ##for j in [pred_label,target_label]:
        for j in range(Ws[nlayer-1].shape[0]):
            m.setObjective(zh[nlayer+1][j], grb.GRB.MAXIMIZE)    
            # m.write('grbtest_LP_2layer_'+str(j)+'.lp')    
            start = time.time()
            m.optimize()
            UB[j] = m.objVal
            m.reset()
            print("[L2][upper bound solved] j = {}, time_lp_solve = {:.4f}".format(j, time.time() - start))
            
        # compute lower bounds        
        ##for j in [pred_label,target_label]:
        for j in range(Ws[nlayer-1].shape[0]):
            m.setObjective(zh[nlayer+1][j], grb.GRB.MINIMIZE)    
            # m.write('grbtest_LP_2layer_'+str(j)+'.lp')    
            start = time.time()
            m.optimize()
            LB[j] = m.objVal
            m.reset()
            print("[L2][lower bound solved] j = {}, time_lp_solve = {:.4f}".format(j, time.time() - start))
            
        bnd_gx0 = LB[target_label]-UB[pred_label]
    else: # use the g_x0 tricks if it's last layer call:
        if untargeted:
            bnd_gx0 = []
            start = time.time()
            for j in range(Ws[nlayer-1].shape[0]):
                if j != pred_label:
                    m.setObjective(zh[nlayer+1][pred_label]-zh[nlayer+1][j], grb.GRB.MINIMIZE)
                    m.optimize()
                    bnd_gx0.append(m.objVal)
                    # print("[L2][Solved untargeted] j = {}, value = {:.4f}".format(j, m.objVal))
                    m.reset()
        else:
            m.setObjective(zh[nlayer+1][pred_label]-zh[nlayer+1][target_label], grb.GRB.MINIMIZE)
            start = time.time()
            m.optimize()
            bnd_gx0 = m.objVal
            m.reset()
        print("[L2][g(x) bound solved] time_lp_solve = {:.4f}".format(time.time() - start))
   
    return UB, LB, bnd_gx0

