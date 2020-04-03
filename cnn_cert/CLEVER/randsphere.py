#!/usr/bin/env python3

import numpy as np
import time
from CLEVER.shmemarray import ShmemRawArray, NpShmemArray
from scipy.special import gammainc
from CLEVER.defense import defend_reduce, defend_jpeg, defend_tv, defend_none, defend_png
from functools import partial

"""
Original Matlab code (for L2 sampling):

function X = randsphere(m,n,r)

X = randn(m,n);
s2 = sum(X.^2,2);
X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);
"""

# generate random signs
def randsign(N):
    n_bytes = (N + 7) // 8
    rbytes = np.random.randint(0, 255, dtype=np.uint8, size=n_bytes)
    return (np.unpackbits(rbytes)[:N] - 0.5) * 2

def l2_samples(m,n):
    X = np.random.randn(m, n)
    s2 = np.sum(X * X, axis = 1)
    return X * (np.tile(1.0*np.power(gammainc(n/2,s2/2), 1/n) / np.sqrt(s2), (n,1))).T

def linf_samples(m, n):
    return np.random.uniform(-1.0, 1.0, (m,n))

def l1_samples(m, n):
    # U is uniform random between 0, 1
    U = np.random.uniform(0, 1.0, (m,n-1))
    V = np.empty(shape=(m,n+1))
    # V is sorted U, with 0 and 1 added to the begin and the end
    V[:,0] = 0.0
    V[:,-1] = 1.0
    V[:,1:-1] = np.sort(U)
    # X is the interval between each V_i
    X = V[:,1:] - V[:,:-1]
    # randomly flip the sign of each X
    s = randsign(m * n).reshape(m,n)
    return X * s

def randsphere(idx, n, r, total_size, scale_size, tag_prefix, input_shape, norm, transform = None):
    # currently we assume r = 1.0 and rescale using the array "scale"
    assert r == 1.0
    result_arr = NpShmemArray(np.float32, (total_size, n), tag_prefix + "randsphere", False)
    # for scale, we may want a different starting point for imagenet, which is scale_start
    scale = NpShmemArray(np.float32, (scale_size, 1), tag_prefix + "scale", False)
    input_example = NpShmemArray(np.float32, input_shape, tag_prefix + "input_example", False)
    all_inputs = NpShmemArray(np.float32, (total_size,) + input_example.shape, tag_prefix + "all_inputs", False)
    # m is the number of items, off is the offset
    m, offset, scale_start = idx
    # n is the dimension
    if norm == "l2":
        samples = l2_samples(m, n)
    if norm == "l1":
        samples = l1_samples(m, n)
    if norm == "li":
        samples = linf_samples(m, n)
    result_arr[offset : offset + m] = samples
    # make a scaling
    result_arr[offset : offset + m] *= scale[offset + scale_start : offset + scale_start + m]
    # add to input example
    all_inputs[offset : offset + m] = input_example
    result_arr = result_arr.reshape(-1, *input_shape)
    all_inputs[offset : offset + m] += result_arr[offset : offset + m]
    # apply an transformation
    if transform:
        transform_new = lambda x: (eval(transform)(np.squeeze(x) + 0.5) - 0.5).reshape(all_inputs[offset].shape)
        # before transformation we need to clip first
        np.clip(all_inputs[offset : offset + m], -0.5, 0.5, out = all_inputs[offset : offset + m])
        # run transformation
        for i in range(offset, offset + m):
            all_inputs[i] = transform_new(all_inputs[i])
        # we need to clip again after transformation (this is in the caller)
    return

