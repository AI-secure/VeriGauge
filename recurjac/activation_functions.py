## activation_functions.py
## 
## Definitions of ReLU, leaky-ReLU and sigmoid family 
## activation functions and their upper and lower bounds
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import numpy as np
from numba import jit

@jit(nopython=True)
def relu_ub_pn(u, l):
    a = u / (u - l)
    return a, -l * a

# upper bound, unsure
@jit(nopython=True)
def leaky_relu_ub_pn(u, l, k):
    a = (u - k * l) / (u - l)
    b = l * u * (k - 1.0) / (u - l)
    return a, b

@jit(nopython=True)
def relu_lb_pn(u, l):
    # adaptive bound
    intercept = np.zeros_like(u)
    slope = np.zeros_like(u)
    mask = np.abs(u) > np.abs(l)
    slope[mask] = 1.0
    return slope, intercept

# lower bound, unsure (adaptive)
@jit(nopython=True)
def leaky_relu_lb_pn(u, l, k):
    # adaptive bound
    intercept = np.zeros_like(u)
    slope = np.full(len(u), k, dtype=u.dtype)
    mask = np.abs(u) > np.abs(l)
    slope[mask] = 1.0
    return slope, intercept

@jit(nopython=True)
def relu_ub_p(u, l):
    return np.ones_like(u), np.zeros_like(u)

# upper bound, positive part
@jit(nopython=True)
def leaky_relu_ub_p(u, l, k):
    return np.ones_like(u), np.zeros_like(u)

@jit(nopython=True)
def relu_lb_p(u, l):
    return np.ones_like(u), np.zeros_like(u)

# lower bound, positive part
@jit(nopython=True)
def leaky_relu_lb_p(u, l, k):
    return np.ones_like(u), np.zeros_like(u)

@jit(nopython=True)
def relu_ub_n(u, l):
    return np.zeros_like(u), np.zeros_like(u)

# upper bound, negative part
@jit(nopython=True)
def leaky_relu_ub_n(u, l, k):
    return np.full(len(u), k, dtype=u.dtype), np.zeros_like(u)

@jit(nopython=True)
def relu_lb_n(u, l):
    return np.zeros_like(u), np.zeros_like(u)

# lower bound, negative part
@jit(nopython=True)
def leaky_relu_lb_n(u, l, k):
    return np.full(len(u), k, dtype=u.dtype), np.zeros_like(u)

@jit(nopython=True)
def act_tanh(y):
    return np.tanh(y)

@jit(nopython=True)
def act_tanh_d(y):
    t = np.cosh(y)
    t = t * t
    return 1.0 / t

@jit(nopython=True)
def act_arctan(y):
    return np.arctan(y)

@jit(nopython=True)
def act_arctan_d(y):
    return 1.0 / (1 + y * y)

@jit(nopython=True)
def act_sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

@jit(nopython=True)
def act_sigmoid_d(y):
    return act_sigmoid(y) * (1 - act_sigmoid(y))

# for I+ case, upper bound
@jit(nopython=True)
def general_ub_n(u, l, func, dfunc):
    alpha = np.empty_like(u)
    mask = np.abs(u-l) > 1e-5
    alpha[mask] = (func(u[mask])-func(l[mask]))/(u[mask]-l[mask])
    mask = np.logical_not(mask)
    alpha[mask] = dfunc(u[mask])
    alpha_UB = alpha;
    beta_UB = func(l) - l * alpha
    return alpha_UB, beta_UB

@jit(nopython=True)
def general_lb_n(u, l, func, dfunc):
    d = 0.5*(u+l);
    alpha_LB = dfunc(d)
    beta_LB = func(d) - d * alpha_LB;
    return alpha_LB, beta_LB

@jit(nopython=True)
def general_ub_p(u, l, func, dfunc):
    d = 0.5*(u+l);
    alpha_UB = dfunc(d)
    beta_UB = func(d) - d * alpha_UB
    return alpha_UB, beta_UB

@jit(nopython=True)
def general_lb_p(u, l, func, dfunc):
    alpha = np.empty_like(u)
    mask = np.abs(u-l) > 1e-5
    alpha[mask] = (func(u[mask])-func(l[mask]))/(u[mask]-l[mask])
    mask = np.logical_not(mask)
    alpha[mask] = dfunc(l[mask])
    beta_LB = func(l) - l * alpha
    alpha_LB = alpha
    return alpha_LB, beta_LB

@jit(nopython=True)
def general_ub_pn(u, l, func, dfunc):
    d_UB = np.empty_like(u)
    for i in range(len(d_UB)):
        d_UB[i] = find_d_UB(u[i],l[i],func,dfunc)
    alpha_UB = (func(d_UB)-func(l))/(d_UB-l)
    beta_UB  = func(l) - (l - 0.01) * alpha_UB
    return alpha_UB, beta_UB

@jit(nopython=True)
def general_lb_pn(u, l, func, dfunc):
    d_LB = np.empty_like(u)
    for i in range(len(d_LB)):
        d_LB[i] = find_d_LB(u[i],l[i],func,dfunc)
    alpha_LB = (func(d_LB)-func(u))/(d_LB-u)
    beta_LB = func(u) - (u + 0.01) * alpha_LB
    return alpha_LB, beta_LB

@jit(nopython=True)
def general_ub_pn_scalar(u, l, func, dfunc):
    d_UB = find_d_UB(u,l,func,dfunc)
    alpha_UB = (func(d_UB)-func(l))/(d_UB-l)
    beta_UB  = func(l) - (l - 0.01) * alpha_UB
    return alpha_UB, beta_UB

@jit(nopython=True)
def general_lb_pn_scalar(u, l, func, dfunc):
    d_LB = find_d_LB(u,l,func,dfunc)
    alpha_LB = (func(d_LB)-func(u))/(d_LB-u)
    beta_LB = func(u) - (u + 0.01) * alpha_LB
    return alpha_LB, beta_LB

@jit(nopython=True)
def find_d_UB(u, l, func, dfunc):
    diff = lambda d,l: (func(d)-func(l))/(d-l) - dfunc(d)
    max_iter = 10;
    d = u/2;
    ub = u; lb = 0;
    for i in range(max_iter):
        t = diff(d, l)
        if t > 0 and np.abs(t) < 0.01:
            break
        if t > 0:
            ub = d;
            d = (d+lb)/2;
        else:
            lb = d;
            d = (d+ub)/2;
    return d

@jit(nopython=True)
def find_d_LB(u,l,func,dfunc):
    diff = lambda d,u: (func(u)-func(d))/(u-d) - dfunc(d)
    max_iter = 10;
    d = l/2;
    ub = 0; lb = l;
    for i in range(max_iter):
        t = diff(d,u)
        if t > 0 and abs(t) < 0.01:
            break
        if t > 0:
            lb = d;
            d = (d+ub)/2;
        else:
            ub = d;
            d = (d+lb)/2;
    return d

def plot_line(u, l, slope, intercept, linestype='--', label='linear'):
    linear_func = lambda x: slope * x + intercept
    plt.plot([l, u], [linear_func(l), linear_func(u)], linestyle=linestype, label=label, marker="o")

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import sys
    matplotlib.rc('font',family='sans-serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 18
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['xtick.labelsize'] = 23
    matplotlib.rcParams['ytick.labelsize'] = 23
    # 'axes.labelsize': 17, 'legend.fontsize': 18,'xtick.labelsize': 16,'ytick.labelsize': 16
    plt.figure(figsize=(6,4))


    if len(sys.argv) == 3:
        l = float(sys.argv[1])
        u = float(sys.argv[2])
    else:
        l = -1.0
        u = 1.75
    assert l < u

    func_name = "tanh"

    if func_name == "tanh":
        func = act_tanh 
        dfunc = act_tanh_d
    elif func_name == "sigmoid":
        func = act_sigmoid 
        dfunc = act_sigmoid_d
    if func_name == "arctan":
        func = act_arctan
        dfunc = act_arctan_d

    x = np.linspace(-3, 3, 1000)
    # plot function
    plt.plot(x, func(x), linestyle='-', label="$\sigma(x)=\\textrm{"+func_name+"}(x)$")
    # get best bounds for func
    if u < 0:
        upper_k, upper_b = general_ub_n(u, l, func, dfunc)
        lower_k, lower_b = general_lb_n(u, l, func, dfunc)
    elif l > 0:
        upper_k, upper_b = general_ub_p(u, l, func, dfunc)
        lower_k, lower_b = general_lb_p(u, l, func, dfunc)
    else:
        upper_k, upper_b = general_ub_pn_scalar(u, l, func, dfunc)
        lower_k, lower_b = general_lb_pn_scalar(u, l, func, dfunc)
        upper_k_test, upper_b_test = general_ub_pn(np.array([u,u]), np.array([l,l]), func, dfunc)
        lower_k_test, lower_b_test = general_lb_pn(np.array([u,u]), np.array([l,l]), func, dfunc)
        assert upper_k_test[0] == upper_k_test[1] == upper_k
        assert lower_k_test[0] == lower_k_test[1] == lower_k

    scale = func(100) - func(-100)
    print(upper_k, upper_b)
    print(lower_k, lower_b)
    plot_line(u, l, upper_k, upper_b, '-.', "Upper Bound")
    plot_line(u, l, lower_k, lower_b, '-.', "Lower Bound")
    plt.plot([u, u], [min(func(-100)-0.1*scale, lower_k*l + lower_b), upper_k*u + upper_b+0.05], linestyle=':', color='gray')
    plt.text(u+0.05, func(-100)-0.075*scale, "$u$", fontsize=23)
    plt.plot([l, l], [min(func(-100)-0.1*scale, lower_k*l + lower_b), upper_k*l + upper_b+0.05], linestyle=':', color='gray')
    plt.text(l+0.05, func(-100)-0.075*scale, "$l$", fontsize=23)


    # plt.xlim(-lim*0.3, lim)
    bottom, top = plt.gca().get_ylim()
    plt.ylim(bottom, top * 1.2)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=6)
    plt.legend()
    plt.tight_layout(h_pad=0.0, w_pad=0.0, pad=0.3)
    plt.savefig('plot_{}_l_{}_u_{}.pdf'.format(func_name, l, u))
    plt.show()

