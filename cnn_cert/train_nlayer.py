#Adapted from https://github.com/huanzhang12/CLEVER 

"""
train_models.py

train the neural network models for attacking

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import argparse
import os
from setup_tinyimagenet import tinyImagenet

# train cnn 7-layer mnist/cifar model
def train_cnn_7layer(data, file_name, params, num_epochs=10, batch_size=256, train_temp=1, init=None, lr=0.01, decay=1e-5, momentum=0.9, activation="relu", optimizer_name="sgd"):
    """
    Train a 7-layer cnn network for MNIST and CIFAR (same as the cnn model in Clever)
    mnist: 32 32 64 64 200 200 
    cifar: 64 64 128 128 256 256
    """

    # create a Keras sequential model
    model = Sequential()

    print("training data shape = {}".format(data.train_data.shape))

    params = [int(p) for p in params]
    # define model structure
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Lambda(tf.atan) if activation == "arctan" else Activation(activation))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Lambda(tf.atan) if activation == "arctan" else Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Lambda(tf.atan) if activation == "arctan" else Activation(activation))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Lambda(tf.atan) if activation == "arctan" else Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Lambda(tf.atan) if activation == "arctan" else Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Lambda(tf.atan) if activation == "arctan" else Activation(activation))
    model.add(Dense(200))

  
    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    if optimizer_name == "sgd":
        # initiate the SGD optimizer with given hyper parameters
        optimizer = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    elif optimizer_name == "adam":
        optimizer = Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay=decay, amsgrad=False)
    
    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    model.summary()
    print("Traing a {} layer model, saving to {}".format(len(params) + 1, file_name))
    # run training with given dataset, and print progress
    history = model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name)
        print('model saved to ', file_name)
    
    return {'model':model, 'history':history}


if __name__ == '__main__':
    
 
    # train tiny imagenet: 1
    #train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_1", params=[32,32,64,64,200,200], num_epochs=30, lr=0.0001, decay=0, activation="relu", optimizer_name="adam")
    
    # train tiny imagenet: 2
    #train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_2", params=[32,32,64,64,200,200], num_epochs=30, lr=0.0005, decay=0, activation="relu", optimizer_name="adam")
    
    # train tiny imagenet: 3
    #train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_3", params=[64,64,64,64,200,200], num_epochs=30, lr=0.0001, decay=0, activation="relu", optimizer_name="adam")
    
    # train tiny imagenet: 4
    #train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_4", params=[100,100,50,50,200,200], num_epochs=30, lr=0.0001, decay=0, activation="relu", optimizer_name="adam")
    
    # train tiny imagenet: 5
    #train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_5", params=[32,32,64,64,200,200], num_epochs=50, lr=0.001, decay=11e-5, activation="relu", optimizer_name="adam")
    
    # train tiny imagenet: 6
    train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_6", params=[32,32,64,64,200,200], num_epochs=60, lr=0.0001, decay=1e-5, activation="relu", optimizer_name="adam")
    
    # train tiny imagenet: 7
    train_cnn_7layer(tinyImagenet(), file_name="models/tiny_cnn_7layer_7", params=[32,32,64,64,200,200], num_epochs=100, lr=0.00005, decay=0, activation="relu", optimizer_name="adam")
    
    
    
    # train mnist
    #train_cnn_7layer(MNIST(), file_name="models/mnist_cnn_7layer", params=[5,5,5,5,5,5], num_epochs=10, lr=0.001, decay=0, activation="relu", optimizer_name="adam")

""" original     
    train_cnn_7layer(MNIST(), file_name="models/mnist_cnn_7layer", params=[32,32,64,64,200,200], num_epochs=10, lr=0.001, decay=0, activation="relu", optimizer_name="adam")
    train_cnn_7layer(MNIST(), file_name="models/mnist_cnn_7layer_sigmoid", params=[32,32,64,64,200,200], num_epochs=10, lr=0.001, decay=0, activation="sigmoid", optimizer_name="adam")
    train_cnn_7layer(MNIST(), file_name="models/mnist_cnn_7layer_tanh", params=[32,32,64,64,200,200], num_epochs=10, lr=0.001, decay=0, activation="tanh", optimizer_name="adam")
    train_cnn_7layer(MNIST(), file_name="models/mnist_cnn_7layer_atan", params=[32,32,64,64,200,200], num_epochs=10, lr=0.001, decay=0, activation="arctan", optimizer_name="adam")
"""   
