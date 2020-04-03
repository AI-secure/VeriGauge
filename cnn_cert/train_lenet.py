#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Adapted from https://github.com/huanzhang12/CLEVER

## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
##                and Huan Zhang <ecezhang@ucdavis.edu>
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None, pool = True):
    """
    Standard neural network training procedure. Trains LeNet-5 style model with pooling optional.
    """
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (5, 5),
                            input_shape=data.train_data.shape[1:], padding='same'))
    model.add(Lambda(tf.nn.relu))
    if pool:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[1], (5, 5)))
    model.add(Lambda(tf.nn.relu))
    if pool:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[2]))
    model.add(Lambda(tf.nn.relu))
    model.add(Dense(10))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return model

    
if not os.path.isdir('models'):
    os.makedirs('models')

if __name__ == '__main__':
    train(MNIST(), "models/mnist_cnn_lenet_nopool", [6, 16, 100], num_epochs=10, pool=False)
    train(MNIST(), "models/mnist_cnn_lenet", [6, 16, 100], num_epochs=10)
