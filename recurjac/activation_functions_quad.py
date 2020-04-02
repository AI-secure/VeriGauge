## activation_functions_qaud.py
## 
## Definitions of ReLU activation function and its quadratic upper and lower bounds
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import numpy as np
from numba import jit

def get_area(a, b, c, lower, upper):
    f = lambda x: 1.0 / 3.0 * a * x * x * x + 1.0 / 2.0 * b * x * x + c * x
    return f(upper) - f(lower)

@jit(nopython=True)
def get_upper_quad_parameterized(u, l, theta):
    a = 1.0 / (u - l) + (1.0 / (u * l)) * theta
    b = (1.0 / (u * l)) * ((l * l * u) / (l - u) - (l + u) * theta)
    c = theta
    return a, b, c

def get_lower_quad_parameterized_lgtu(u, l, theta):
    a = theta / (u * u - l * u)
    b = -a * l
    c = 0
    print("u={:.3f}, l={:.3f}, theta={:.3f}, a={:.3f}, b={:.3f}, c={:.3f}, ".format(u, l, theta, a, b, c), end='')
    area1 = - get_area(a, b, c, l, 0)
    area2 = 0.5*u*u - get_area(a, b, c, 0, u)
    area = area1 + area2
    area_other = 0.5*u*u - theta / (u*u - l*u) * (1.0 / 6.0 * l*l*l + 1.0 / 3.0 * u*u*u - 0.5 * l * u*u)
    print("area1={:.3f}, area2={:.3f}, area={:.3f} ({:.3f}), reference={:.3f}".format(area1, area2, area, area_other, 0.5*u*u))
    return a, b, c

def get_lower_quad_parameterized_lltu(u, l, theta):
    a = (l - theta) / (u*l - l*l)
    b = (u*theta - l*l) / (u*l - l*l)
    c = 0
    print("u={:.3f}, l={:.3f}, theta={:.3f}, a={:.3f}, b={:.3f}, c={:.3f}, ".format(u, l, theta, a, b, c), end='')
    area1 = - get_area(a, b, c, l, 0)
    area2 = 0.5*u*u - get_area(a, b, c, 0, u)
    area = area1 + area2
    area_other = 0.5*u*u + 1.0/(u*l-l*l)*(1.0/3.0*(l-theta)*l*l*l + 1.0/2.0*(u*theta-l*l)*l*l - 1.0/3.0*(l-theta)*u*u*u - 0.5*(u*theta-l*l)*u*u)
    print("area1={:.3f}, area2={:.3f}, area={:.3f} ({:.3f}), reference={:.3f}".format(area1, area2, area, area_other, 0.5*l*l))
    return a, b, c

@jit(nopython=True)
def get_lower_quad_parameterized(u, l, theta):
    a = theta / (u * u - l * u)
    b = -a * l
    c = 0
    """
    print("u={:.3f}, l={:.3f}, theta={:.3f}, a={:.3f}, b={:.3f}, c={:.3f}, ".format(u, l, theta, a, b, c), end='')
    area1 = - get_area(a, b, c, l, 0)
    area2 = 0.5*u*u - get_area(a, b, c, 0, u)
    area = area1 + area2
    area_other = 0.5*u*u - theta / (u*u - l*u) * (1.0 / 6.0 * l*l*l + 1.0 / 3.0 * u*u*u - 0.5 * l * u*u)
    print("area1={:.3f}, area2={:.3f}, area={:.3f} ({:.3f}), reference={:.3f}".format(area1, area2, area, area_other, min(0.5*u*u, 0.5*l*l)))
    """
    return a, b, c

@jit(nopython=True)
def get_best_lower_quad(u, l):
    if abs(l) >= abs(u):
        return 0, 0, 0
        if abs(l) >= 2.0 * abs(u):
            return 0, 0, 0
        else:
            return get_lower_quad_parameterized(u, l, u)
    else:
        # return 0, 1, 0
        if abs(u) >= 2.0 * abs(l):
            return 0, 1, 0
        else:
            return get_lower_quad_parameterized(u, l, u)
        
@jit(nopython=True)
def get_best_upper_quad(u, l):
    if abs(l) > abs(u):
        bias = l / (l - u) + 2 * l / (u - l)
        k = 2.0 / u - (l + u) / (u * l)
        theta = - bias / k
    else:
        bias = l / (l - u) + 2 * u / (u - l)
        k = 2.0 / l - (l + u) / (u * l)
        theta = (1.0 - bias) / k
    # print("l = {}, u = {}, bias = {}, k = {}, theta = {}".format(l, u, bias, k, theta))
    # assert theta >= 0
    # assert theta <= (- u * l) / (u - l)
    a, b, c = get_upper_quad_parameterized(u, l, theta)
    # print("slope at l: {}".format(2 * a * l + b))
    # print("slope at u: {}".format(2 * a * u + b))
    return a, b, c

def plot_parameterized(func, u, l, theta, label, linestyle='--'):
    x = np.linspace(l * 1.3, u * 1.3, 1000)
    a, b, c = func(u, l, theta)
    y = a * x**2 + b*x + c
    plt.plot(x, y, linestyle=linestyle, label=label)

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import sys

    if len(sys.argv) == 3:
        l = float(sys.argv[1])
        u = float(sys.argv[2])
    else:
        l = -1.0
        u = 1.75
    assert l <= 0
    assert u >= 0

    x = np.linspace(l * 1.3, u * 1.3, 1000)

    # plot relu
    plt.plot([l,0,u], [0, 0, u], linestyle='--', label="ReLU")
    # get best upper quad bound
    a, b, c = get_best_upper_quad(u, l)
    y = a * x**2 + b*x + c
    best_theta = c
    upper = (- u * l) / (u - l)
    print("valid theta: 0 - {}, best {}".format(upper, best_theta))
    lower = c / 2
    plt.plot(x, y, linestyle='-.', label="Best upper")
    # slope u/(u-l) linear lower bound
    plt.plot([l, 0, u], [l*u / (u - l), 0, u * u / (u - l)], linestyle=':', label="linear lower")
    # slope u/(u-l) linear lower bound
    plt.plot([l, 0, u], [0, -l*u / (u - l), u], linestyle=':', label="linear upper")
    """
    for theta in np.linspace(lower, upper, 3):
        if abs(theta - best_theta) < 1e-5:
            continue
        a, b, c = get_upper_quad_parameterized(u, l, theta)
        y = a * x**2 + b*x + c
        plt.plot(x, y, label="theta = {}".format(theta))
    """
    if abs(u) < abs(l):
        for theta in np.linspace(0, u, 10):
            plot_parameterized(get_lower_quad_parameterized_lgtu, u, l, theta, "theta={}".format(theta))
    else:
        for theta in np.linspace(l, 0, 10):
            plot_parameterized(get_lower_quad_parameterized_lltu, u, l, theta, "theta={}".format(theta))

    # get best lower quad bound
    a, b, c = get_best_lower_quad(u, l)
    y = a * x**2 + b*x + c
    plt.plot(x, y, linestyle='-.', label="Best lower")

    lim = max(abs(l), abs(u))
    plt.xlim(-lim, lim)
    plt.ylim(-lim*0.3, lim)
    plt.legend()
    plt.show()

