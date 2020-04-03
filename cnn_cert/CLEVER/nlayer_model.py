## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2017-2018, IBM Corp.
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
##
## This program is licenced under the Apache 2.0 licence,
## contained in the LICENCE file in this directory.

import numpy as np
import os
import pickle
import gzip
import argparse
import urllib.request

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K


class NLayerModel:
    def __init__(self, params, restore = None, session=None, use_log=False, image_size=28, image_channel=1):
        
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
            model.add(Activation('relu'))
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
    args = parser.parse_args()
    nlayers = len(args.layer_parameters) + 1

    import tensorflow as tf
    with tf.Session() as sess:
        # if a model file is not specified, use a manual override
        if not args.modelfile:
            args.modelfile = "models/"+args.model+"_"+str(nlayers)+"layer_relu"
        if args.model == "mnist":
            model =  NLayerModel(args.layer_parameters, args.modelfile, sess)
            #model =  NLayerModel(args.layer_parameters, "models/mnist_"+str(nlayers)+"layer_relu")
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

