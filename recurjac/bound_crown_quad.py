## bound_crown_quad.py
## 
## Implementation of CROWN-quad bound
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import numpy as np
from numba import jit
import recurjac.activation_functions_quad as quad_fit
# import .activation_functions_quad as quad_fit

@jit(nopython=True)
def proj_li(x, i_lb, i_ub):
    return np.minimum(np.maximum(x, i_lb), i_ub)

@jit(nopython=True)
def proj_l2(x, x0, eps):
    delta = x - x0
    norm = np.linalg.norm(delta)
    if norm > eps:
        delta /= (norm / eps)
    return x0 + delta

@jit(nopython=True)
def proj_l1(x_real, x0, q):
    x = x_real - x0
    # sort the array in descending order
    s = np.sort(np.abs(x))[::-1]
    cs = np.cumsum(s)
    r = np.arange(1, len(s)+1)
    s2 = np.empty_like(s)
    s2[:-1] = s[1:]
    s2[-1] = 0
    t = cs - r * s2 
    ndxs = np.nonzero(t >= q + 1e-8)[0]
    if len(ndxs):
        ndx = ndxs[0]
        thresh = np.float32((cs[ndx] - q) / (ndx+1))
        x = x * (np.float32(1) - thresh / np.maximum( np.abs(x), thresh))
    return x + x0


# quadratic bounds, two layers only
# currently we only use the bounds of the first two layers
@jit(nopython=True)
def crown_quad_bound(Ws,bs,UBs,LBs,neuron_state,nlayer,x0,eps,p_np,verbose=False):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1)
    # for 2-layer, we can use other norms (for input perturbation x)
    assert p_np == np.inf or nlayer == 2

    x0 = x0.astype(np.float32)

    # we only consider the last two layers
    W2 = Ws[nlayer-1]
    W1 = Ws[nlayer-2]
    b2 = bs[nlayer-1]
    b1 = bs[nlayer-2]
    # upper and lower bounds of hidden layer
    h_ub = UBs[nlayer-1]
    h_lb = LBs[nlayer-1]
    # upper and lower bounds of input layer
    if nlayer == 2:
        i_ub = UBs[nlayer-2]
        i_lb = LBs[nlayer-2]
    else:
        # UBs and LBs are pre-relu values, apply ReLU here
        i_ub = np.maximum(UBs[nlayer-2], 0)
        i_lb = np.maximum(LBs[nlayer-2], 0)

    UB_final = np.empty_like(b2)
    LB_final = np.empty_like(b2)

    # indices for unsure neurons
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]
    idx_active = np.nonzero(neuron_state[nlayer - 2] == 1)[0]
    # form linear term slope
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    alpha[idx_unsure] = UBs[nlayer-1][idx_unsure]/(UBs[nlayer-1][idx_unsure] - LBs[nlayer-1][idx_unsure])
    # quadratic bound parameters
    a = np.empty_like(b1)
    b = np.empty_like(b1)
    # fit best quadratic lower bound
    for r in idx_unsure:
        a[r], b[r], _ = quad_fit.get_best_lower_quad(h_ub[r], h_lb[r])

    for j in range(len(b2)):
        # positive entries in j-th row, unsure neurons
        pos = np.nonzero(W2[j][idx_unsure] > 0)[0]
        # negative entries in j-th row, unsure neurons
        neg = np.nonzero(W2[j][idx_unsure] < 0)[0]
        # unsure neurons, corresponding to positive entries in W2
        idx_unsure_pos = idx_unsure[pos]
        # unsure neurons, corresponding to negative entries in W2
        idx_unsure_neg = idx_unsure[neg]

        # form the diag matrix for quadratic term
        diag_ub = np.zeros_like(b1)
        diag_lb = np.zeros_like(b1)
        # all diagnal elements are guaranteed negative
        diag_ub[idx_unsure_neg] = W2[j][idx_unsure_neg] * a[idx_unsure_neg]
        diag_lb[idx_unsure_pos] = W2[j][idx_unsure_pos] * a[idx_unsure_pos]
        # form the vector for linear term
        lamb_ub = np.zeros_like(b1)
        lamb_lb = np.zeros_like(b1)
        lamb_ub[idx_active] = W2[j][idx_active]
        lamb_ub[idx_unsure_pos] = W2[j][idx_unsure_pos]*alpha[idx_unsure_pos]
        lamb_ub[idx_unsure_neg] = W2[j][idx_unsure_neg]*b[idx_unsure_neg]
        lamb_lb[idx_active] = W2[j][idx_active]
        lamb_lb[idx_unsure_neg] = W2[j][idx_unsure_neg]*alpha[idx_unsure_neg]
        lamb_lb[idx_unsure_pos] = W2[j][idx_unsure_pos]*b[idx_unsure_pos]
        # form the constant term
        const_ub = -np.dot(W2[j][idx_unsure_pos] * alpha[idx_unsure_pos], h_lb[idx_unsure_pos])
        const_lb = -np.dot(W2[j][idx_unsure_neg] * alpha[idx_unsure_neg], h_lb[idx_unsure_neg])
        const_ub += b2[j]
        const_lb += b2[j]

        # expand to previous layer
        # add more constant terms
        const_ub += np.dot(b1, diag_ub * b1) + np.dot(lamb_ub, b1)
        const_lb += np.dot(b1, diag_lb * b1) + np.dot(lamb_lb, b1)
        # quadratic term
        # Q_ub = np.dot(W1.T * diag_ub, W1)
        # Q_lb = np.dot(W1.T * diag_lb, W1)
        # form only the non-zero part
        diag_ub_part = diag_ub[idx_unsure_neg[a[idx_unsure_neg] != 0]]
        Q_half_ub = (W1[idx_unsure_neg[a[idx_unsure_neg] != 0]].T * np.sqrt(-diag_ub_part)).T
        diag_lb_part = diag_lb[idx_unsure_pos[a[idx_unsure_pos] != 0]]
        Q_half_lb = (W1[idx_unsure_pos[a[idx_unsure_pos] != 0]].T * np.sqrt(diag_lb_part)).T
        # linear term
        B_ub = 2 * np.dot(W1.T * diag_ub, b1) + np.dot(W1.T, lamb_ub)
        B_lb = 2 * np.dot(W1.T * diag_lb, b1) + np.dot(W1.T, lamb_lb)
        # print("number of components: ub, lb", np.sum(a[idx_unsure_neg] != 0), np.sum(a[idx_unsure_pos] != 0))
        # print("Q.shape, B.shape:", Q_ub.shape, B_ub.shape)
        # print("Q_half.shape, diag_part.shape", Q_half_ub.shape, diag_ub_part.shape)

        # w, v = np.linalg.eig(Q_ub)
        # print(w[:20])
        # print("const_ub, const_lb:", const_ub, const_lb)

        x = (i_lb + i_ub)
        x /= 2.0
        f_qp_lb = lambda x : np.dot(x, np.dot(Q_half_lb.T, np.dot(Q_half_lb, x))) + np.dot(B_lb, x) + const_lb
        # f_qp_lb = lambda x : np.dot(x, np.dot(Q_lb, x)) + np.dot(B_lb, x) + const_lb
        # obj = np.dot(x, np.dot(Q_lb, x)) + np.dot(B_lb, x) + const_lb
        grad_lb = lambda x : np.dot(Q_half_lb.T, np.dot(Q_half_lb, x)) + B_lb
        # grad_lb = lambda x : np.dot(Q_lb, x) + B_lb
        # grad = np.dot(Q_lb, x) + B_lb
        # proj = lambda x : np.minimum(np.maximum(x, i_lb), i_ub)

        # for L1 norm, use a larger eta
        eta_init = np.float32(1.0)
        if p_np == 1:
            eta_init = np.float32(10.0)

        q_np = 1
        if p_np == 1:
            q_np = np.inf
        if p_np == 2:
            q_np = 2
        # no quadratic term, no need to run
        if len(Q_half_lb) == 0:
            LB_final[j] = -np.linalg.norm(B_lb, q_np) * eps + np.dot(B_lb, x) + const_lb
            converged_lb = True
        else:
            last_obj = np.inf
            eta = eta_init
            obj = f_qp_lb(x)
            converged_lb = False
            for i in range(200):
                if abs(last_obj - obj) < 1e-4:
                    if verbose:
                        print("solver terminated, obj = ", obj)
                    converged_lb = True
                    break
                cur_grad = grad_lb(x)
                
                if p_np == np.inf:
                    proj_grad = proj_li(x - cur_grad, i_lb, i_ub)
                elif p_np == 2:
                    proj_grad = proj_l2(x - cur_grad, x0, eps)
                elif p_np == 1:
                    proj_grad = proj_l1(x - cur_grad, x0, eps)
                proj_grad -= x

                grad_norm = np.linalg.norm(proj_grad)
                eta = eta_init
                # line search
                while eta > 1e-5:
                    if p_np == np.inf:
                        new_grad = proj_li(x - eta * cur_grad, i_lb, i_ub)
                    elif p_np == 2:
                        new_grad = proj_l2(x - eta * cur_grad, x0, eps)
                    elif p_np == 1:
                        new_grad = proj_l1(x - eta * cur_grad, x0, eps)
                    new_grad_norm = np.dot(new_grad - x, proj_grad)
                    if not (f_qp_lb(new_grad) > obj - np.float32(0.1) * eta * new_grad_norm):
                        break
                    eta *= np.float32(0.5)
                x -= eta * cur_grad

                if p_np == np.inf:
                    x = proj_li(x, i_lb, i_ub)
                elif p_np == 2:
                    x = proj_l2(x, x0, eps)
                elif p_np == 1:
                    x = proj_l1(x, x0, eps)

                last_obj = obj 
                obj = f_qp_lb(x)
                if verbose:
                    print("iter", i, "eta", eta, "obj", obj, "norm", np.linalg.norm(x - x0, p_np))
            LB_final[j] = obj

        x = (i_lb + i_ub)
        x /= 2.0
        f_qp_ub = lambda x : -np.dot(x, np.dot(Q_half_ub.T, np.dot(Q_half_ub, x))) + np.dot(B_ub, x) + const_ub
        # obj = np.dot(x, np.dot(Q_ub, x)) + np.dot(B_ub, x) + const_ub
        grad_ub = lambda x : -np.dot(Q_half_ub.T, np.dot(Q_half_ub, x)) + B_ub
        # grad = np.dot(Q_ub, x) + B_ub

        # no quadratic term, no need to run
        if len(Q_half_ub) == 0:
            UB_final[j] = np.linalg.norm(B_ub, q_np) * eps + np.dot(B_ub, x) + const_ub
            converged_ub = True
        else:
            last_obj = -np.inf
            eta = eta_init
            obj = f_qp_ub(x)
            converged_ub = True
            for i in range(200):
                if abs(last_obj - obj) < 1e-4:
                    if verbose:
                        print("solver terminated, obj = ", obj)
                    converged_ub = True
                    break
                cur_grad = grad_ub(x)

                if p_np == np.inf:
                    proj_grad = proj_li(x - cur_grad, i_lb, i_ub)
                elif p_np == 2:
                    proj_grad = proj_l2(x - cur_grad, x0, eps)
                elif p_np == 1:
                    proj_grad = proj_l1(x - cur_grad, x0, eps)
                proj_grad -= x

                grad_norm = np.linalg.norm(proj_grad)
                eta = eta_init
                # line search
                while eta > 1e-5:
                    if p_np == np.inf:
                        new_grad = proj_li(x + eta * cur_grad, i_lb, i_ub)
                    elif p_np == 2:
                        new_grad = proj_l2(x + eta * cur_grad, x0, eps)
                    elif p_np == 1:
                        new_grad = proj_l1(x + eta * cur_grad, x0, eps)
                    new_grad_norm = np.dot(new_grad - x, proj_grad)
                    if not (f_qp_ub(new_grad) < obj + np.float32(0.1) * eta * new_grad_norm):
                        break
                    eta *= np.float32(0.5)
                x += eta * cur_grad
                if p_np == np.inf:
                    x = proj_li(x, i_lb, i_ub)
                elif p_np == 2:
                    x = proj_l2(x, x0, eps)
                elif p_np == 1:
                    x = proj_l1(x, x0, eps)
                last_obj = obj 
                obj = f_qp_ub(x)
                if verbose:
                    x = proj_l1(x, x0, eps)
                    print("iter", i, "eta", eta, "obj", obj, "norm", np.linalg.norm(x - x0, p_np))
            UB_final[j] = obj
        print(j+1, "/", len(b2), "solved, size", len(diag_ub_part), len(diag_lb_part), ", converged", converged_lb and converged_ub)

    return UB_final, LB_final

# quadratic bounds, two layers only
# currently we only use the bounds of the last two layers
@jit(nopython=True)
def get_layer_bound_quad_both(Ws,bs,UBs,LBs,neuron_state,nlayer,x0,eps,p_n):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1)

    # we only consider the last two layers
    W2 = Ws[nlayer-1]
    W1 = Ws[nlayer-2]
    b2 = bs[nlayer-1]
    b1 = bs[nlayer-2]
    # upper and lower bounds of hidden layer
    h_ub = UBs[nlayer-1]
    h_lb = LBs[nlayer-1]
    # upper and lower bounds of input layer
    if nlayer == 2:
        i_ub = UBs[nlayer-2]
        i_lb = LBs[nlayer-2]
    else:
        # UBs and LBs are pre-relu values, apply ReLU here
        i_ub = np.maximum(UBs[nlayer-2], 0)
        i_lb = np.maximum(LBs[nlayer-2], 0)

    UB_final = np.empty_like(b2)
    LB_final = np.empty_like(b2)

    # indices for unsure neurons
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]
    idx_active = np.nonzero(neuron_state[nlayer - 2] == 1)[0]
    # form linear term slope
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    alpha[idx_unsure] = UBs[nlayer-1][idx_unsure]/(UBs[nlayer-1][idx_unsure] - LBs[nlayer-1][idx_unsure])
    # quadratic bound parameters
    a_low = np.empty_like(b1)
    b_low = np.empty_like(b1)
    a_high = np.empty_like(b1)
    b_high = np.empty_like(b1)
    c_high = np.empty_like(b1)
    # fit best quadratic lower bound
    for r in idx_unsure:
        a_r, b_r, c_r = quad_fit.get_best_lower_quad(h_ub[r], h_lb[r])
        a_low[r] = a_r
        b_low[r] = b_r
        a_r, b_r, c_r = quad_fit.get_best_upper_quad(h_ub[r], h_lb[r])
        a_high[r] = a_r
        b_high[r] = b_r
        c_high[r] = c_r

    for j in range(len(b2)):
        # positive entries in j-th row, unsure neurons
        pos = np.nonzero(W2[j][idx_unsure] > 0)[0]
        # negative entries in j-th row, unsure neurons
        neg = np.nonzero(W2[j][idx_unsure] < 0)[0]
        # unsure neurons, corresponding to positive entries in W2
        idx_unsure_pos = idx_unsure[pos]
        # unsure neurons, corresponding to negative entries in W2
        idx_unsure_neg = idx_unsure[neg]

        # form the diag matrix for quadratic term
        diag_ub = np.zeros_like(b1)
        diag_lb = np.zeros_like(b1)
        # all diagnal elements are guaranteed negative
        diag_ub[idx_unsure_neg] = W2[j][idx_unsure_neg] * a_low[idx_unsure_neg]
        diag_ub[idx_unsure_pos] = W2[j][idx_unsure_pos] * a_high[idx_unsure_pos]
        diag_lb[idx_unsure_pos] = W2[j][idx_unsure_pos] * a_low[idx_unsure_pos]
        diag_lb[idx_unsure_neg] = W2[j][idx_unsure_neg] * a_high[idx_unsure_neg]
        # form the vector for linear term
        lamb_ub = np.zeros_like(b1)
        lamb_lb = np.zeros_like(b1)
        lamb_ub[idx_active] = W2[j][idx_active]
        lamb_ub[idx_unsure_pos] = W2[j][idx_unsure_pos]*b_high[idx_unsure_pos]
        lamb_ub[idx_unsure_neg] = W2[j][idx_unsure_neg]*b_low[idx_unsure_neg]
        lamb_lb[idx_active] = W2[j][idx_active]
        lamb_lb[idx_unsure_neg] = W2[j][idx_unsure_neg]*b_high[idx_unsure_neg]
        lamb_lb[idx_unsure_pos] = W2[j][idx_unsure_pos]*b_low[idx_unsure_pos]
        # form the constant term
        const_ub = np.dot(W2[j][idx_unsure_pos], c_high[idx_unsure_pos])
        const_lb = np.dot(W2[j][idx_unsure_neg], c_high[idx_unsure_neg])
        const_ub += b2[j]
        const_lb += b2[j]

        # expand to previous layer
        # add more constant terms
        const_ub += np.dot(b1, diag_ub * b1) + np.dot(lamb_ub, b1)
        const_lb += np.dot(b1, diag_lb * b1) + np.dot(lamb_lb, b1)
        # quadratic term
        Q_ub = np.dot(W1.T * diag_ub, W1)
        Q_lb = np.dot(W1.T * diag_lb, W1)
        # form only the non-zero part
        diag_ub_part = diag_ub[idx_unsure]
        Q_half_ub = W1[idx_unsure]
        diag_lb_part = diag_lb[idx_unsure]
        Q_half_lb = W1[idx_unsure]
        # linear term
        B_ub = 2 * np.dot(W1.T * diag_ub, b1) + np.dot(W1.T, lamb_ub)
        B_lb = 2 * np.dot(W1.T * diag_lb, b1) + np.dot(W1.T, lamb_lb)
        # print("number of components: ub, lb", np.sum(a[idx_unsure_neg] != 0), np.sum(a[idx_unsure_pos] != 0))
        # print("Q.shape, B.shape:", Q_ub.shape, B_ub.shape)
        # print("Q_half.shape, diag_part.shape", Q_half_ub.shape, diag_ub_part.shape)

        # w, v = np.linalg.eig(Q_ub)
        # print(w[:20])
        # print("const_ub, const_lb:", const_ub, const_lb)

        x = (i_lb + i_ub)
        x /= 2.0
        # x = i_lb
        f_qp_lb = lambda x : np.dot(x, np.dot(Q_half_lb.T * diag_lb_part, np.dot(Q_half_lb, x))) + np.dot(B_lb, x) + const_lb
        # obj = np.dot(x, np.dot(Q_lb, x)) + np.dot(B_lb, x) + const_lb
        grad_lb = lambda x : np.dot(Q_half_lb.T * diag_lb_part, np.dot(Q_half_lb, x)) + B_lb
        # grad = np.dot(Q_lb, x) + B_lb
        proj = lambda x : np.minimum(np.maximum(x, i_lb), i_ub)
        last_obj = np.inf
        eta = np.float32(1.0)
        obj = f_qp_lb(x)
        for i in range(200):
            if abs(last_obj - obj) < 1e-4:
                print("solver terminated, obj = ", obj)
                break
            cur_grad = grad_lb(x)
            grad_norm = np.linalg.norm(cur_grad)
            eta = np.float32(1.0)
            # line search
            while f_qp_lb(proj(x - eta * cur_grad)) > obj - np.float32(0.1) * eta * grad_norm:
                eta *= np.float32(0.5)
            x -= eta * cur_grad
            x = proj(x)
            last_obj = obj 
            obj = f_qp_lb(x)
            print("iter ", i, " eta ", eta, " obj ", obj)
        LB_final[j] = obj

        x = (i_lb + i_ub)
        x /= 2.0
        # x = i_lb
        f_qp_ub = lambda x : -np.dot(x, np.dot(Q_half_ub.T * diag_ub_part, np.dot(Q_half_ub, x))) + np.dot(B_ub, x) + const_ub
        # obj = np.dot(x, np.dot(Q_ub, x)) + np.dot(B_ub, x) + const_ub
        grad_ub = lambda x : -np.dot(Q_half_ub.T * diag_ub_part, np.dot(Q_half_ub, x)) + B_ub
        # grad = np.dot(Q_ub, x) + B_ub
        last_obj = -np.inf
        eta = np.float32(1.0)
        obj = f_qp_ub(x)
        for i in range(200):
            if abs(last_obj - obj) < 1e-4:
                print("solver terminated, obj = ", obj)
                break
            cur_grad = grad_ub(x)
            grad_norm = np.linalg.norm(cur_grad)
            eta = np.float32(1.0)
            # line search
            while f_qp_ub(proj(x + eta * cur_grad)) < obj + np.float32(0.1) * eta * grad_norm:
                eta *= np.float32(0.5)
            x += eta * cur_grad
            x = proj(x)
            last_obj = obj 
            obj = f_qp_ub(x)
            print("iter ", i, " eta ", eta, " obj ", obj)
        UB_final[j] = obj

    return UB_final, LB_final
