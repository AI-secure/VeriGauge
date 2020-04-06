"""
cnn_bounds_full.py

Main CNN-Cert computation file for general networks

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
from numba import njit, jit
import numpy as np

from .setup_mnist import MNIST
from .setup_cifar import CIFAR
from .setup_tinyimagenet import tinyImagenet
from .cnn_bounds_full_core import pool, conv, conv_bound, conv_full, conv_bound_full, pool_linear_bounds

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from .train_resnet import ResidualStart, ResidualStart2
import tensorflow as tf
from .utils import generate_data
import time
from .activations import relu_linear_bounds, ada_linear_bounds, atan_linear_bounds, sigmoid_linear_bounds, tanh_linear_bounds
linear_bounds = None

import random

def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
#General model class
class Model:
    def __init__(self, model, inp_shape = (28,28,1)):
        temp_weights = [layer.get_weights() for layer in model.layers]

        self.shapes = []
        self.sizes = []
        self.weights = []
        self.biases = []
        self.pads = []
        self.strides = []
        self.types = []
        self.model = model
        
        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        i = 0
        while i < len(model.layers):
            layer = model.layers[i]
            i += 1
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv2D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtype=np.float32)
                else:
                    W, b = weights
                    W = W.astype(np.float32)
                    b = b.astype(np.float32)
                padding = layer.get_config()['padding']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+W.shape[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+W.shape[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-W.shape[1])/stride[1])+1, W.shape[-1])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == GlobalAveragePooling2D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[0],cur_shape[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(cur_shape[0]*cur_shape[1])
                pad = (0,0,0,0)
                stride = ((1,1))
                cur_shape = (1,1,cur_shape[2])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == AveragePooling2D:
                print('avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                padding = layer.get_config()['padding']
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                W = np.zeros((pool_size[0],pool_size[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(pool_size[0]*pool_size[1])
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Activation or type(layer) == Lambda:
                print('activation')
                self.types.append('relu')
                self.sizes.append(None)
                self.strides.append(None)
                self.pads.append(None)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == BatchNormalization:
                print('batch normalization')
                gamma, beta, mean, std = weights
                std = np.sqrt(std**2+0.001) #Avoids zero division
                a = gamma/std
                b = -gamma*mean/std+beta
                self.weights[-1] = a*self.weights[-1]
                self.biases[-1] = a*self.biases[-1]+b
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                b = b.astype(np.float32)
                W = W.reshape(list(cur_shape)+[W.shape[-1]]).astype(np.float32)
                cur_shape = (1,1,W.shape[-1])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append((1,1))
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Dropout:
                print('dropout')
            elif type(layer) == MaxPooling2D:
                print('pool')
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                padding = layer.get_config()['padding']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.types.append('pool')
                self.sizes.append(pool_size)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            elif type(layer) == ResidualStart2:
                print('basic block 2')
                conv1 = model.layers[i]
                bn1 = model.layers[i+1]
                conv2 = model.layers[i+3]
                conv3 = model.layers[i+4]
                bn2 = model.layers[i+5]
                bn3 = model.layers[i+6]
                i = i+8

                W1, bias1 = conv1.get_weights()
                W2, bias2 = conv2.get_weights()
                W3, bias3 = conv3.get_weights()
                
                gamma1, beta1, mean1, std1 = bn1.get_weights()
                std1 = np.sqrt(std1**2+0.001) #Avoids zero division
                a1 = gamma1/std1
                b1 = gamma1*mean1/std1+beta1
                W1 = a1*W1
                bias1 = a1*bias1+b1
                
                gamma2, beta2, mean2, std2 = bn2.get_weights()
                std2 = np.sqrt(std2**2+0.001) #Avoids zero division
                a2 = gamma2/std2
                b2 = gamma2*mean2/std2+beta2
                W2 = a2*W2
                bias2 = a2*bias2+b2
                 
                gamma3, beta3, mean3, std3 = bn3.get_weights()
                std3 = np.sqrt(std3**2+0.001) #Avoids zero division
                a3 = gamma3/std3
                b3 = gamma3*mean3/std3+beta3
                W3 = a3*W3
                bias3 = a3*bias3+b3

                padding1 = conv1.get_config()['padding']
                stride1 = conv1.get_config()['strides']
                pad1 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding1 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride1[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride1[1]))
                    total_padding_h = stride1[0]*(desired_h-1)+W1.shape[0]-cur_shape[0]
                    total_padding_w = stride1[1]*(desired_w-1)+W1.shape[1]-cur_shape[1]
                    pad1 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad1[0]+pad1[1]-W1.shape[0])/stride1[0])+1, int((cur_shape[1]+pad1[2]+pad1[3]-W1.shape[1])/stride1[1])+1, W1.shape[3])

                padding2 = conv2.get_config()['padding']
                stride2 = conv2.get_config()['strides']
                pad2 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding2 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride2[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride2[1]))
                    total_padding_h = stride2[0]*(desired_h-1)+W2.shape[0]-cur_shape[0]
                    total_padding_w = stride2[1]*(desired_w-1)+W2.shape[1]-cur_shape[1]
                    pad2 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))

                padding3 = conv3.get_config()['padding']
                stride3 = conv3.get_config()['strides']
                pad3 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding3 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride3[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride3[1]))
                    total_padding_h = stride3[0]*(desired_h-1)+W3.shape[0]-cur_shape[0]
                    total_padding_w = stride3[1]*(desired_w-1)+W3.shape[1]-cur_shape[1]
                    pad3 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))

                W1 = np.ascontiguousarray(W1.transpose((3,0,1,2)).astype(np.float32))
                bias1 = np.ascontiguousarray(bias1.astype(np.float32))
                W2 = np.ascontiguousarray(W2.transpose((3,0,1,2)).astype(np.float32))
                bias2 = np.ascontiguousarray(bias2.astype(np.float32))
                W3 = np.ascontiguousarray(W3.transpose((3,0,1,2)).astype(np.float32))
                bias3 = np.ascontiguousarray(bias3.astype(np.float32))
                self.types.append('basic_block_2')
                self.sizes.append(None)
                self.strides.append((stride1, stride2, stride3))
                self.pads.append((pad1, pad2, pad3))
                self.shapes.append(cur_shape)
                self.weights.append((W1, W2, W3))
                self.biases.append((bias1, bias2, bias3))
            elif type(layer) == ResidualStart:
                print('basic block')
                conv1 = model.layers[i]
                bn1 = model.layers[i+1]
                conv2 = model.layers[i+3]
                bn2 = model.layers[i+4]
                i = i+6

                W1, bias1 = conv1.get_weights()
                W2, bias2 = conv2.get_weights()
                
                gamma1, beta1, mean1, std1 = bn1.get_weights()
                std1 = np.sqrt(std1**2+0.001) #Avoids zero division
                a1 = gamma1/std1
                b1 = gamma1*mean1/std1+beta1
                W1 = a1*W1
                bias1 = a1*bias1+b1
                
                gamma2, beta2, mean2, std2 = bn2.get_weights()
                std2 = np.sqrt(std2**2+0.001) #Avoids zero division
                a2 = gamma2/std2
                b2 = gamma2*mean2/std2+beta2
                W2 = a2*W2
                bias2 = a2*bias2+b2

                padding1 = conv1.get_config()['padding']
                stride1 = conv1.get_config()['strides']
                pad1 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding1 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride1[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride1[1]))
                    total_padding_h = stride1[0]*(desired_h-1)+W1.shape[0]-cur_shape[0]
                    total_padding_w = stride1[1]*(desired_w-1)+W1.shape[1]-cur_shape[1]
                    pad1 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad1[0]+pad1[1]-W1.shape[0])/stride1[0])+1, int((cur_shape[1]+pad1[2]+pad1[3]-W1.shape[1])/stride1[1])+1, W1.shape[3])

                padding2 = conv2.get_config()['padding']
                stride2 = conv2.get_config()['strides']
                pad2 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding2 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride2[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride2[1]))
                    total_padding_h = stride2[0]*(desired_h-1)+W2.shape[0]-cur_shape[0]
                    total_padding_w = stride2[1]*(desired_w-1)+W2.shape[1]-cur_shape[1]
                    pad2 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))

                W1 = np.ascontiguousarray(W1.transpose((3,0,1,2)).astype(np.float32))
                bias1 = np.ascontiguousarray(bias1.astype(np.float32))
                W2 = np.ascontiguousarray(W2.transpose((3,0,1,2)).astype(np.float32))
                bias2 = np.ascontiguousarray(bias2.astype(np.float32))
                self.types.append('basic_block')
                self.sizes.append(None)
                self.strides.append((stride1, stride2))
                self.pads.append((pad1, pad2))
                self.shapes.append(cur_shape)
                self.weights.append((W1, W2))
                self.biases.append((bias1, bias2))
            else:
                print(str(type(layer)))
                raise ValueError('Invalid Layer Type')
        print(cur_shape)

    def predict(self, data):
        return self.model(data)


@njit
def UL_conv_bound(A, B, pad, stride, shape, W, b, inner_pad, inner_stride, inner_shape):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = B.copy()
    assert A.shape[5] == W.shape[0]
                                                

    for x in range(A_new.shape[0]):
        p_start = np.maximum(0, pad[0]-stride[0]*x)
        p_end = np.minimum(A.shape[3], shape[0]+pad[0]-stride[0]*x)
        t_start = np.maximum(0, -stride[0]*inner_stride[0]*x+inner_stride[0]*pad[0]+inner_pad[0])
        t_end = np.minimum(A_new.shape[3], inner_shape[0]-stride[0]*inner_stride[0]*x+inner_stride[0]*pad[0]+inner_pad[0])
        for y in range(A_new.shape[1]):
            q_start = np.maximum(0, pad[2]-stride[1]*y)
            q_end = np.minimum(A.shape[4], shape[1]+pad[2]-stride[1]*y)
            u_start = np.maximum(0, -stride[1]*inner_stride[1]*y+inner_stride[1]*pad[2]+inner_pad[2])
            u_end = np.minimum(A_new.shape[4], inner_shape[1]-stride[1]*inner_stride[1]*y+inner_stride[1]*pad[2]+inner_pad[2])
            for t in range(t_start, t_end):
                for u in range(u_start, u_end):
                    for p in range(p_start, p_end):
                        for q in range(q_start, q_end):
                            if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2]:
                                A_new[x,y,:,t,u,:] += np.dot(A[x,y,:,p,q,:],W[:,t-inner_stride[0]*p,u-inner_stride[1]*q,:])
            for p in range(p_start, p_end):
                for q in range(q_start, q_end):
                    B_new[x,y,:] += np.dot(A[x,y,:,p,q,:],b)
    return A_new, B_new

basic_block_2_cache = {}
def UL_basic_block_2_bound(A, B, pad, stride, W1, W2, W3, b1, b2, b3, pad1, pad2, pad3, stride1, stride2, stride3, upper=True):
    LB, UB = basic_block_2_cache[np.sum(W1)]
    A1, B1 = UL_conv_bound(A, B, np.asarray(pad), np.asarray(stride), np.asarray(UB.shape), W2, b2, np.asarray(pad2), np.asarray(stride2), np.asarray(UB.shape))
    inter_pad = (stride2[0]*pad[0]+pad2[0], stride2[0]*pad[1]+pad2[1], stride2[1]*pad[2]+pad2[2], stride2[1]*pad[3]+pad2[3])
    inter_stride = (stride2[0]*stride[0], stride2[1]*stride[1])
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    if upper:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_u, alpha_l, beta_u, beta_l)
    else:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_l, alpha_u, beta_l, beta_u)
    A1, B1 = UL_conv_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), np.asarray(UB.shape), W1, b1, np.asarray(pad1), np.asarray(stride1), np.asarray(UB.shape))
    A2, B2 = UL_conv_bound(A, B, np.asarray(pad), np.asarray(stride), np.asarray(UB.shape), W3, b3, np.asarray(pad3), np.asarray(stride3), np.asarray(UB.shape))
    height_diff = A1.shape[3]-A2.shape[3]
    width_diff = A1.shape[4]-A2.shape[4]
    assert height_diff % 2 == 0
    assert width_diff % 2 == 0
    d_h = height_diff//2
    d_w = width_diff//2
    A1[:,:,:,d_h:A1.shape[3]-d_h,d_w:A1.shape[4]-d_w,:] += A2
    return A1, B1+B2-B

basic_block_cache = {}
def UL_basic_block_bound(A, B, pad, stride, W1, W2, b1, b2, pad1, pad2, stride1, stride2, upper=True):
    LB, UB = basic_block_cache[np.sum(W1)]
    A1, B1 = UL_conv_bound(A, B, np.asarray(pad), np.asarray(stride), np.asarray(UB.shape), W2, b2, np.asarray(pad2), np.asarray(stride2), np.asarray(UB.shape))
    inter_pad = (stride2[0]*pad[0]+pad2[0], stride2[0]*pad[1]+pad2[1], stride2[1]*pad[2]+pad2[2], stride2[1]*pad[3]+pad2[3])
    inter_stride = (stride2[0]*stride[0], stride2[1]*stride[1])
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    if upper:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_u, alpha_l, beta_u, beta_l)
    else:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_l, alpha_u, beta_l, beta_u)
    A1, B1 = UL_conv_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), np.asarray(UB.shape), W1, b1, np.asarray(pad1), np.asarray(stride1), np.asarray(UB.shape))
    height_diff = A1.shape[3]-A.shape[3]
    width_diff = A1.shape[4]-A.shape[4]
    assert height_diff % 2 == 0
    assert width_diff % 2 == 0
    d_h = height_diff//2
    d_w = width_diff//2
    A1[:,:,:,d_h:A1.shape[3]-d_h,d_w:A1.shape[4]-d_w,:] += A
    return A1, B1

@njit
def UL_relu_bound(A, B, pad, stride, alpha_u, alpha_l, beta_u, beta_l):
    A_new = np.zeros_like(A)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    B_new = B.copy()
    for x in range(A_new.shape[0]):
        p_start = np.maximum(0, pad[0]-stride[0]*x)
        p_end = np.minimum(A.shape[3], alpha_u.shape[0]+pad[0]-stride[0]*x)
        for y in range(A_new.shape[1]):
            q_start = np.maximum(0, pad[2]-stride[1]*y)
            q_end = np.minimum(A.shape[4], alpha_u.shape[1]+pad[2]-stride[1]*y)
            for z in range(A_new.shape[2]):
                for p in range(p_start, p_end):
                    for q in range(q_start, q_end):
                        for r in range(A.shape[5]):
                            A_new[x,y,z,p,q,r] += A_plus[x,y,z,p,q,r]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
                            A_new[x,y,z,p,q,r] += A_minus[x,y,z,p,q,r]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
                            B_new[x,y,z] += A_plus[x,y,z,p,q,r]*beta_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
                            B_new[x,y,z] += A_minus[x,y,z,p,q,r]*beta_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
    return A_new, B_new

@njit
def UL_pool_bound(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, alpha_u, alpha_l, beta_u, beta_l):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = B.copy()
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    inner_index_y = u+stride[1]*inner_stride[1]*x-inner_stride[1]*pad[2]-inner_pad[2]
                    if 0<=inner_index_x<inner_shape[0] and 0<=inner_index_y<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and 0<=u-inner_stride[1]*q<alpha_u.shape[1] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[2] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[3]:
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new += conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)
    return A_new, B_new

#Main function to find bounds at each layer
def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, pads, strides, sizes, types, LBs, UBs):
    if types[nlayer-1] == 'relu':
        #print('relu')
        return np.maximum(LBs[nlayer-1], 0), np.maximum(UBs[nlayer-1], 0)
    elif types[nlayer-1] == 'conv':
        #print('conv')
        A_u = weights[nlayer-1].reshape((1, 1, weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2], weights[nlayer-1].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2], weights[nlayer-1].shape[3]), dtype=np.float32)
        B_u = biases[nlayer-1]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = pads[nlayer-1]
        stride = strides[nlayer-1]
    elif types[nlayer-1] == 'pool':
        #print('pool')
        A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = (1,1)
        alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LBs[nlayer-1], UBs[nlayer-1], np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]),  np.asarray(sizes[nlayer-1]))
        A_u, B_u = UL_pool_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(sizes[nlayer-1]), np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]), np.asarray(LBs[nlayer-1].shape), alpha_u, alpha_l, beta_u, beta_l)
        A_l, B_l = UL_pool_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(sizes[nlayer-1]), np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]), np.asarray(LBs[nlayer-1].shape), alpha_l, alpha_u, beta_l, beta_u)
        pad = pads[nlayer-1]
        stride = strides[nlayer-1]
    elif types[nlayer-1] == 'basic_block_2':
        #print('basic block 2')
        W1, W2, W3 = weights[nlayer-1]
        b1, b2, b3 = biases[nlayer-1]
        pad1, pad2, pad3 = pads[nlayer-1]
        stride1, stride2, stride3 = strides[nlayer-1]
        LB, UB = compute_bounds(weights[:nlayer-1]+[W1], biases[:nlayer-1]+[b1], out_shape, nlayer, x0, eps, p_n, pads[:nlayer-1]+[pad1], strides[:nlayer-1]+[stride1], sizes, types[:nlayer-1]+['conv'], LBs, UBs)
        basic_block_2_cache[np.sum(W1)] = (LB, UB)

        A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = (1,1)
        A_u, B_u = UL_basic_block_2_bound(A_u, B_u, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=True)
        A_l, B_l = UL_basic_block_2_bound(A_l, B_l, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=False)
        inner_pad = pad3
        inner_stride = stride3
        pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
        stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
    elif types[nlayer-1] == 'basic_block':
        #print('basic block')
        W1, W2 = weights[nlayer-1]
        b1, b2 = biases[nlayer-1]
        pad1, pad2 = pads[nlayer-1]
        stride1, stride2 = strides[nlayer-1]
        LB, UB = compute_bounds(weights[:nlayer-1]+[W1], biases[:nlayer-1]+[b1], out_shape, nlayer, x0, eps, p_n, pads[:nlayer-1]+[pad1], strides[:nlayer-1]+[stride1], sizes, types[:nlayer-1]+['conv'], LBs, UBs)
        basic_block_cache[np.sum(W1)] = (LB, UB)

        A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = (1,1)
        A_u, B_u = UL_basic_block_bound(A_u, B_u, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=True)
        A_l, B_l = UL_basic_block_bound(A_l, B_l, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=False)
    
    for i in range(nlayer-2, -1, -1):
        if types[i] == 'conv':
            #print('conv')
            A_u, B_u = UL_conv_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(UBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape))
            A_l, B_l = UL_conv_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(LBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape))
            pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
            stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
        elif types[i] == 'pool':
            #print('pool')
            alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LBs[i], UBs[i], np.asarray(pads[i]), np.asarray(strides[i]),  np.asarray(sizes[i]))
            A_u, B_u = UL_pool_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(sizes[i]), np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape), alpha_u, alpha_l, beta_u, beta_l)
            A_l, B_l = UL_pool_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(sizes[i]), np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape), alpha_l, alpha_u, beta_l, beta_u)
            pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
            stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
        elif types[i] == 'relu':
            #print('relu')
            alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LBs[i], UBs[i])
            A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), alpha_u, alpha_l, beta_u, beta_l)
            A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), alpha_l, alpha_u, beta_l, beta_u)
        elif types[i] == 'basic_block_2':
            #print('basic block 2')
            A_u, B_u = UL_basic_block_2_bound(A_u, B_u, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=True)
            A_l, B_l = UL_basic_block_2_bound(A_l, B_l, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=False)
            inner_pad = pads[i][0]
            inner_stride = strides[i][0]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
            inner_pad = pads[i][1]
            inner_stride = strides[i][1]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
        elif types[i] == 'basic_block':
            #print('basic block')
            A_u, B_u = UL_basic_block_bound(A_u, B_u, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=True)
            A_l, B_l = UL_basic_block_bound(A_l, B_l, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=False)
            inner_pad = pads[i][0]
            inner_stride = strides[i][0]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
            inner_pad = pads[i][1]
            inner_stride = strides[i][1]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
    LUB, UUB = conv_bound_full(A_u, B_u, np.asarray(pad), np.asarray(stride), x0, eps, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, np.asarray(pad), np.asarray(stride), x0, eps, p_n)
    return LLB, UUB

#Main function to find output bounds
def find_output_bounds(weights, biases, shapes, pads, strides, sizes, types, x0, eps, p_n, param_linear_bounds):
    global linear_bounds
    linear_bounds = param_linear_bounds
    LBs = [x0-eps]
    UBs = [x0+eps]
    for i in range(1,len(weights)+1):
        # print('Layer ' + str(i))
        LB, UB = compute_bounds(weights, biases, shapes[i], i, x0, eps, p_n, pads, strides, sizes, types, LBs, UBs)
        UBs.append(UB)
        LBs.append(LB)
    return LBs[-1], UBs[-1]

#Warms up numba functions
def warmup(model, x, eps_0, p_n, func):
    print('Warming up...')
    weights = model.weights[:-1]
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    last_weight = np.ascontiguousarray((W[0,:,:,:]).reshape([1]+list(W.shape[1:])),dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1,1,1))
    func(weights, biases, shapes, model.pads, model.strides, model.sizes, model.types, x, eps_0, p_n)

#Main function to compute CNN-Cert bound
def run(file_name, n_samples, p_n, q_n, activation = 'relu', cifar=False, tinyimagenet=False):
    np.random.seed(1215)
    tf.set_random_seed(1215)
    random.seed(1215)
    keras_model = load_model(file_name, custom_objects={'fn':loss, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf})
    if tinyimagenet:
        model = Model(keras_model, inp_shape = (64,64,3))
    elif cifar:
        model = Model(keras_model, inp_shape = (32,32,3))
    else:
        model = Model(keras_model)

    #Set correct linear_bounds function
    global linear_bounds
    if activation == 'relu':
        linear_bounds = relu_linear_bounds
    elif activation == 'ada':
        linear_bounds = ada_linear_bounds
    elif activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'arctan':
        linear_bounds = atan_linear_bounds
    
    if cifar:
        inputs, targets, true_labels, true_ids, img_info = generate_data(CIFAR(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    elif tinyimagenet:
        inputs, targets, true_labels, true_ids, img_info = generate_data(tinyImagenet(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)
    else:
        inputs, targets, true_labels, true_ids, img_info = generate_data(MNIST(), samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0)

    if len(inputs) == 0:
        return 0, 0
    
    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2
    #0b0100 <- least
    preds = model.model.predict(inputs[0][np.newaxis,:]).flatten()
    steps = 15
    eps_0 = 0.05
    summation = 0

    warmup(model, inputs[0].astype(np.float32), eps_0, p_n, find_output_bounds)
        
    start_time = time.time()
    for i in range(len(inputs)):
        print('--- CNN-Cert: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        weights = model.weights[:-1]
        biases = model.biases[:-1]
        shapes = model.shapes[:-1]
        W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
        last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
        weights.append(last_weight)
        biases.append(np.asarray([b[predict_label]-b[target_label]]))
        shapes.append((1,1,1))

        #Perform binary searchcc
        log_eps = np.log(eps_0)
        log_eps_min = -np.inf
        log_eps_max = np.inf
        for j in range(steps):
            #print('Step ' + str(j))
            LB, UB = find_output_bounds(weights, biases, shapes, model.pads, model.strides, model.sizes, model.types, inputs[i].astype(np.float32), np.exp(log_eps), p_n)
            
            print("Step {}, eps = {:.5f}, {:.6s} <= f_c - f_t <= {:.6s}".format(j,np.exp(log_eps),str(np.squeeze(LB)),str(np.squeeze(UB))))
            if LB > 0: #Increase eps
                log_eps_min = log_eps
                log_eps = np.minimum(log_eps+1, (log_eps_max+log_eps_min)/2)
            else: #Decrease eps
                log_eps_max = log_eps
                log_eps = np.maximum(log_eps-1, (log_eps_max+log_eps_min)/2)
       
        if p_n == 105:
            str_p_n = 'i'
        else:
            str_p_n = str(p_n)

        print("[L1] method = CNN-Cert-{}, model = {}, image no = {}, true_id = {}, target_label = {}, true_label = {}, norm = {}, robustness = {:.5f}".format(activation,file_name, i, true_ids[i],target_label,predict_label,str_p_n,np.exp(log_eps_min)))
        summation += np.exp(log_eps_min)
    K.clear_session()
    
    eps_avg = summation/len(inputs)
    total_time = (time.time()-start_time)/len(inputs)
    print("[L0] method = CNN-Cert-{}, model = {}, total images = {}, norm = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(activation,file_name,len(inputs),str_p_n,eps_avg,total_time))
    return eps_avg, total_time
    
