## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import os
import pickle
import gzip
import argparse
import urllib.request

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
import tensorflow as tf


class NLayerModel:
    def __init__(self, params, restore = None, session=None, use_log=False, image_size=28, image_channel=1, activation='relu'):
        
        self.image_size = image_size
        self.num_channels = image_channel
        self.num_labels = 10
        
        model = Sequential()
        model.add(Flatten(input_shape=(image_size, image_size, image_channel)))
        # list of all hidden units weights
        self.U = []
        for param in params:
            # add each dense layer, and save a reference to list U
            self.U.append(Dense(param))
            model.add(self.U[-1])
            # ReLU activation
            # model.add(Activation(activation))
            if activation == "arctan":
                model.add(Lambda(lambda x: tf.atan(x)))
            else:
                model.add(Activation(activation))
        self.W = Dense(10)
        model.add(self.W)
        # output log probability, used for black-box attack
        if use_log:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.layer_outputs = layer_outputs
        self.model = model

    def predict(self, data):
        return self.model(data)

def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

class CNNModel:
    def __init__(self, file_name, inp_shape = (28,28,1)):
        model = load_model(file_name, custom_objects={'fn':loss, 'tf':tf})
        temp_weights = [layer.get_weights() for layer in model.layers]

        self.weights = []
        self.biases = []
        self.model = model
        
        cur_shape = inp_shape
        i = 0
        while i < len(model.layers):
            layer = model.layers[i]
            i += 1
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


                if type(model.layers[i+1]) == BatchNormalization:
                    print('batch normalization')
                    gamma, beta, mean, std = weights
                    std = np.sqrt(std**2+0.001) #Avoids zero division
                    aa = gamma/std
                    bb = gamma*mean/std+beta
                    W = aa*W
                    b = aa*b+bb
                    i += 1

                new_shape = (cur_shape[0]-W.shape[0]+1, cur_shape[1]-W.shape[1]+1, W.shape[-1])

                flat_inp = np.prod(cur_shape)
                flat_out = np.prod(new_shape)
                W_flat = np.zeros((flat_inp, flat_out))
                b_flat = np.zeros((flat_out))
                m,n,p = cur_shape
                d,e,f = new_shape
                for x in range(d):
                    for y in range(e):
                        for z in range(f):
                            b_flat[e*f*x+f*y+z] = b[z]
                            for k in range(p):
                                for idx0 in range(W.shape[0]):
                                    for idx1 in range(W.shape[1]):
                                        ii = idx0 + x
                                        jj = idx1 + y
                                        W_flat[n*p*ii+p*jj+k, e*f*x+f*y+z]=W[idx0, idx1, k, z]
                
                cur_shape = new_shape
                self.weights.append(W_flat.T)
                self.biases.append(b_flat.T)
            elif type(layer) == Activation:
                print('activation')
            elif type(layer) == Lambda:
	            print('lambda')
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                self.weights.append(W.T)
                self.biases.append(b.T)
            elif type(layer) == BatchNormalization:
                print('batch normalization')
            elif type(layer) == Dropout:
                print('dropout')
            elif type(layer) == MaxPooling2D:
                print('pool')
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            else:
                print(str(type(layer)))
                raise ValueError('Invalid Layer Type')
        print(cur_shape)

    def predict(self, data):
        return self.model(data)


if __name__ == "__main__":
    import scipy.io as sio
    parser = argparse.ArgumentParser(description='save n-layer MNIST and CIFAR weights')
    parser.add_argument('--model', 
                default="mnist",
                choices=["mnist", "cifar"],
                help='model name')
    parser.add_argument('--modelfile', 
                default="",
                help='override the model filename, use user specied one')
    parser.add_argument('layer_parameters',
                nargs='+',
                help='number of hidden units per layer')
    parser.add_argument('--activation',
                default="relu",
                choices=["relu", "tanh", "sigmoid", "elu", "hard_sigmoid", "softplus"])
    args = parser.parse_args()
    nlayers = len(args.layer_parameters) + 1

    import tensorflow as tf
    with tf.Session() as sess:
        # if a model file is not specified, use a manual override
        if not args.modelfile:
            args.modelfile = "models/"+args.model+"_"+str(nlayers)+"layer_"+args.activation
        if args.model == "mnist":
            model =  NLayerModel(args.layer_parameters, args.modelfile, sess)
            #model =  NLayerModel(args.layer_parameters, "models/mnist_"+str(nlayers)+"layer_"+args.activation)
        elif args.model == "cifar":
            model =  NLayerModel(args.layer_parameters, args.modelfile, sess, image_size=32, image_channel=3)
        else:
            raise(RuntimeError("Unknow model"))

        
        [W, bias_W] = model.W.get_weights()
        save_dict = {'W': W, 'bias_W': bias_W}
        print("Output layer shape:", W.shape)
        U = model.U
        for i, Ui in enumerate(U):
            # save hidden layer weights, layer by layer
            [weight_Ui, bias_Ui] = Ui.get_weights()
            print("Hidden layer {} shape: {}".format(i, weight_Ui.shape))
            save_dict['U'+str(i+1)] = weight_Ui
            save_dict['bias_U'+str(i+1)] = bias_Ui

        save_name = args.model + "_" + str(nlayers) + "layers"
        print('saving to {}.mat with matrices {}'.format(save_name, save_dict.keys()))
        # results saved to mnist.mat or cifar.mat
        sio.savemat(save_name, save_dict)

