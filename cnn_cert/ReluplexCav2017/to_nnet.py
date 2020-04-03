from tensorflow.contrib.keras.api.keras.models import load_model

import numpy as np
from setup_mnist import MNIST
import tensorflow as tf

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

def get_weights_biases(file_name):
    model = load_model(file_name, custom_objects={'fn':fn})
    temp_weights = [layer.get_weights() for layer in model.layers]
    weights = []
    biases = []
    for i in range(len(temp_weights)):
        if i % 2 != 0:
            W = temp_weights[i][0].T
            weights.append(W)
            biases.append(temp_weights[i][1])
    return weights, biases
    

def nnet(file_name, suffix = ''):
    weights, biases = get_weights_biases(file_name)
    nlayers = len(weights)
    sizes = [weights[0].shape[1]]
    for W in weights:
        sizes.append(W.shape[0])
    with open(file_name+suffix+'.nnet', 'w') as f:
        l1 = [str(nlayers),str(sizes[0]),str(sizes[-1]),str(max(sizes))]
        print(','.join(l1), file=f)
        str_sizes = [str(s) for s in sizes]
        print(','.join(str_sizes), file=f)
        print(0, file=f)
        print(','.join(['-0.5' for i in range(sizes[0])]), file=f)
        print(','.join(['0.5' for i in range(sizes[0])]), file=f)
        print(','.join(['0.0' for i in range(sizes[0]+1)]), file=f)
        print(','.join(['1.0' for i in range(sizes[0]+1)]), file=f)
        for j in range(len(weights)):
            W, b = weights[j], biases[j]
            if j == len(weights)-1:
                W = -W
                b = -b
            for i in range(W.shape[0]):
                l = list(W[i,:])
                l = [str(x) for x in l]
                print(','.join(l), file=f)
            for i in range(b.shape[0]):
                print(str(b[i]), file=f)


















    
