#!/usr/bin/env python3
## train_nlayer.py
## 
## Train a n-layer network with different activation functions for experiments
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
from mnist_cifar_models import NLayerModel
import argparse
import os

def train(data, file_name, params, num_epochs=50, batch_size=256, train_temp=1, init=None, lr=0.01, decay=0.0, momentum=0.9, activation="relu", activation_param=None, grad_reg = 0.0, dropout_rate = 0.0):
    """
    Train a n-layer simple network for MNIST and CIFAR
    """
    
    # create a Keras sequential model
    model = NLayerModel(params, use_softmax=False, image_size=data.train_data.shape[1], image_channel=data.train_data.shape[3], activation =activation, activation_param = activation_param, l2_reg = decay, dropout_rate = dropout_rate)
    model = model.model

    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)
        grad = tf.gradients(loss, model.input)[0]
        grad_norm = tf.reduce_sum(tf.square(grad), axis = (1,2,3))
        return loss + grad_reg * grad_norm

    # initiate the SGD optimizer with given hyper parameters
    # sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    sgd = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    
    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    # model.summary()
    print("Traing a {} layer model, saving to {}".format(len(params) + 1, file_name))
    # save model to a file
    if file_name != None:
        checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # run training with given dataset, and print progress
    history = model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              callbacks=[checkpoint],
              shuffle=True)
    
    return {'model':model, 'history':history}

if not os.path.isdir('models'):
    os.makedirs('models')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train n-layer MNIST and CIFAR models')
    parser.add_argument('--model', 
                default="mnist",
                choices=["mnist", "cifar"],
                help='model name')
    parser.add_argument('--modelfile', 
                default="",
                help='override the model filename, use user specied one')
    parser.add_argument('--modelpath', 
                default="models_training",
                help='folder for saving trained models')
    parser.add_argument('layer_parameters',
                nargs='+',
                help='number of hidden units per layer')
    parser.add_argument('--activation',
                default="relu",
                choices=["relu", "tanh", "sigmoid", "arctan", "elu", "hard_sigmoid", "softplus", "leaky"])
    parser.add_argument('--leaky_slope',
                type=float,
                default=0.3)
    parser.add_argument('--lr',
                default=0.01,
                type=float,
                help='learning rate')
    parser.add_argument('--wd',
                default=0.0,
                type=float,
                help='weight decay')
    parser.add_argument('--dropout',
                default=0.0,
                type=float,
                help='dropout rate')
    parser.add_argument('--gradreg',
                default=0.0,
                type=float,
                help='gradient regularization')
    parser.add_argument('--epochs',
                default=50,
                type=int,
                help='number of epochs')
    parser.add_argument('--overwrite',
                action='store_true',
                help='overwrite output file')
    args = parser.parse_args()
    print(args)
    nlayers = len(args.layer_parameters) + 1
    if not args.modelfile:
        file_name = args.modelpath+"/"+args.model+"_"+str(nlayers)+"layer_"+args.activation+"_"+args.layer_parameters[0]
    else:
        file_name = args.modelfile
    print("Model will be saved to", file_name)
    if os.path.isfile(file_name) and not args.overwrite:
        raise RuntimeError("model {} exists.".format(file_name))
    if args.model == "mnist":
        data = MNIST()
    elif args.model == "cifar":
        data = CIFAR()
    train(data, file_name=file_name, params=args.layer_parameters, num_epochs=args.epochs, lr=args.lr, decay=args.wd, activation=args.activation, activation_param=args.leaky_slope, grad_reg=args.gradreg, dropout_rate = args.dropout)

