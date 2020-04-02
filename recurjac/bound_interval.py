## bound_crown.py
## 
## Implementation of interval bound propagation (IBP)
##
## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

from numba import jit, njit
import numpy as np

@njit
def interval_bound(W_Nk,b_Nk,UB_prev,LB_prev,x0,eps,p_n):

    if p_n == np.inf:
        # for Linf norm we consider bounds for each individual element and do not use eps
        gamma = np.empty_like(W_Nk)
        eta = np.empty_like(gamma)
        UB_Nk = np.empty_like(b_Nk)
        LB_Nk = np.empty_like(b_Nk)
        
        for ii in range(W_Nk.shape[0]):
            for jj in range(W_Nk.shape[1]):
                if W_Nk[ii,jj] > 0:
                    gamma[ii,jj] = UB_prev[jj]
                    eta[ii,jj] = LB_prev[jj]
                else:
                    gamma[ii,jj] = LB_prev[jj]
                    eta[ii,jj] = UB_prev[jj]
                  
            UB_Nk[ii] = np.dot(W_Nk[ii], gamma[ii])+b_Nk[ii]
            LB_Nk[ii] = np.dot(W_Nk[ii], eta[ii])+b_Nk[ii]

        return UB_Nk, LB_Nk
   
    else:
        Ax0 = np.dot(W_Nk,x0)
        UB_first = np.empty_like(b_Nk)
        LB_first = np.empty_like(b_Nk)
        # dual norm for all other norms
        q_n = int(1.0/ (1.0 - 1.0/p_n)) if p_n != 1 else np.inf
        for j in range(W_Nk.shape[0]):
            dualnorm_Aj = np.linalg.norm(W_Nk[j], q_n)
            UB_first[j] = Ax0[j]+eps*dualnorm_Aj+b_Nk[j]
            LB_first[j] = Ax0[j]-eps*dualnorm_Aj+b_Nk[j]

        return UB_first, LB_first

