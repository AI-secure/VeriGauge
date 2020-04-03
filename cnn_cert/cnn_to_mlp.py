"""
cnn_to_mlp.py

Converts CNNs to MLP networks

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""

from tensorflow.keras.models import load_model
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from tensorflow.contrib.keras.api.keras.callbacks import LambdaCallback
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
from tensorflow.contrib.keras.api.keras import backend as K

import numpy as np
from setup_mnist import MNIST
from setup_cifar import CIFAR
import tensorflow as tf

import time as timing
import datetime

ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
#Prints to log file
def printlog(s):
    print(s, file=open("log_cnn2mlp_"+timestr+".txt", "a"))

#Function to get weights from saved model
def get_weights(file_name, inp_shape=(28,28,1)):
    model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})
    temp_weights = [layer.get_weights() for layer in model.layers]
    new_params = []
    eq_weights = []
    cur_size = inp_shape
    for p in temp_weights:
        if len(p) > 0:
            W, b = p
            eq_weights.append([])
            if len(W.shape) == 2: #FC
                eq_weights.append([W, b])
            else: # Conv
                new_size = (cur_size[0]-W.shape[0]+1, cur_size[1]-W.shape[1]+1, W.shape[-1])
                flat_inp = np.prod(cur_size)
                flat_out = np.prod(new_size)
                new_params.append(flat_out)
                W_flat = np.zeros((flat_inp, flat_out))
                b_flat = np.zeros((flat_out))
                m,n,p = cur_size
                d,e,f = new_size
                for x in range(d):
                    for y in range(e):
                        for z in range(f):
                            b_flat[e*f*x+f*y+z] = b[z]
                            for k in range(p):
                                for idx0 in range(W.shape[0]):
                                    for idx1 in range(W.shape[1]):
                                        i = idx0 + x
                                        j = idx1 + y
                                        W_flat[n*p*i+p*j+k, e*f*x+f*y+z]=W[idx0, idx1, k, z]
                eq_weights.append([W_flat, b_flat])
                cur_size = new_size
    print('Weights found')
    return eq_weights, new_params

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

#Main function to convert CNN to MLP model
def convert(file_name, new_name, cifar = False):
    if not cifar:
        eq_weights, new_params = get_weights(file_name)
        data = MNIST()
    else:
        eq_weights, new_params = get_weights(file_name, inp_shape = (32,32,3))
        data = CIFAR()
    model = Sequential()
    model.add(Flatten(input_shape=data.train_data.shape[1:]))
    for param in new_params:
        model.add(Dense(param))
        model.add(Lambda(lambda x: tf.nn.relu(x)))
    model.add(Dense(10))
    
    for i in range(len(eq_weights)):
        try:
            print(eq_weights[i][0].shape)
        except:
            pass
        model.layers[i].set_weights(eq_weights[i])

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.save(new_name)
    acc = model.evaluate(data.validation_data, data.validation_labels)[1]
    printlog("Converting CNN to MLP")
    nlayer = file_name.split('_')[-3][0]
    filters = file_name.split('_')[-2]
    kernel_size = file_name.split('_')[-1]
    printlog("model name = {0}, numlayer = {1}, filters = {2}, kernel size = {3}".format(file_name,nlayer,filters,kernel_size))
    printlog("Model accuracy: {:.3f}".format(acc))
    printlog("-----------------------------------")
    return acc

if __name__ == '__main__':
    table = 3
    printlog("-----------------------------------")
    if table == 3 or table == 4:
        #Table 3+4
        convert('models/mnist_cnn_4layer_5_3', 'models/mnist_cnn_as_mlp_4layer_5_3')
        convert('models/mnist_cnn_4layer_20_3', 'models/mnist_cnn_as_mlp_4layer_20_3')
        convert('models/mnist_cnn_5layer_5_3', 'models/mnist_cnn_as_mlp_5layer_5_3')
        convert('models/cifar_cnn_7layer_5_3', 'models/cifar_cnn_as_mlp_7layer_5_3', cifar=True)
        convert('models/cifar_cnn_5layer_10_3', 'models/cifar_cnn_as_mlp_5layer_10_3', cifar=True)
    
    if table == 10 or table == 11:
        #Table 10+11
        convert('models/mnist_cnn_2layer_5_3', 'models/mnist_cnn_as_mlp_2layer_5_3')
        convert('models/mnist_cnn_3layer_5_3', 'models/mnist_cnn_as_mlp_3layer_5_3')
        convert('models/mnist_cnn_6layer_5_3', 'models/mnist_cnn_as_mlp_6layer_5_3')
        convert('models/mnist_cnn_7layer_5_3', 'models/mnist_cnn_as_mlp_7layer_5_3')
        convert('models/mnist_cnn_8layer_5_3', 'models/mnist_cnn_as_mlp_8layer_5_3')
        convert('models/cifar_cnn_5layer_5_3', 'models/cifar_cnn_as_mlp_5layer_5_3', cifar=True)
        convert('models/cifar_cnn_6layer_5_3', 'models/cifar_cnn_as_mlp_6layer_5_3', cifar=True)
        convert('models/cifar_cnn_8layer_5_3', 'models/cifar_cnn_as_mlp_8layer_5_3', cifar=True)
        convert('models/mnist_cnn_4layer_10_3', 'models/mnist_cnn_as_mlp_4layer_10_3')
        convert('models/mnist_cnn_8layer_10_3', 'models/mnist_cnn_as_mlp_8layer_10_3')
        convert('models/cifar_cnn_7layer_10_3', 'models/cifar_cnn_as_mlp_7layer_10_3', cifar=True)
        convert('models/mnist_cnn_8layer_20_3', 'models/mnist_cnn_as_mlp_8layer_20_3')
        convert('models/cifar_cnn_5layer_20_3', 'models/cifar_cnn_as_mlp_5layer_20_3', cifar=True)
        convert('models/cifar_cnn_7layer_20_3', 'models/cifar_cnn_as_mlp_7layer_20_3', cifar=True)


