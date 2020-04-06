## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]
        print(" ========= data type ============")
        print("data type = {}".format(self.test_data))
        

class MNISTModel:
    def __init__(self, restore = None, session=None, use_log=False, use_brelu = False):
        def bounded_relu(x):
                return K.relu(x, max_value=1)
        if use_brelu:
            activation = bounded_relu
        else:
            activation = 'relu'
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(Activation(activation))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dense(10))
        # output log probability, used for black-box attack
        if use_log:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.model = model
        self.layer_outputs = layer_outputs

    def predict(self, data):
        return self.model(data)

class TwoLayerMNISTModel:
    def __init__(self, restore = None, session=None, use_log=False):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(1024))
        model.add(Lambda(lambda x: x * 10))
        model.add(Activation('softplus'))
        model.add(Lambda(lambda x: x * 0.1))
        model.add(Dense(10))
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

class MadryMNISTModel(object):

    class PredictModel(object):
        def __init__(self, sess, predict_gen):
            self.input = None
            self.output = None
            self.sess = sess
            self.predict_gen = predict_gen

        def predict(self, data):
            if self.input is None:
                print("creating a new graph for inference")
                self.input = tf.placeholder(dtype=tf.float32, shape = [None, 28, 28, 1])
                self.output = self.predict_gen(self.input, "Inference_MadryMNIST")
            return self.sess.run([self.output], feed_dict = {self.input: data})
            

    def __init__(self, restore = None, session=None, use_log=False):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.sess = session
        self.restore = restore
        self.use_log = use_log
        self.model_file = tf.train.latest_checkpoint(restore)
        if self.model_file is None:
            raise(FileNotFoundError("model directory " + restore + " is invalid"))
        self.model = self.PredictModel(self.sess, self.predict)

    def predict(self, data, name_prefix = "MadryMNIST"):
        with tf.name_scope(name_prefix):
            # keep a record of the variables we created
            start_vars = set(x.name for x in tf.global_variables())

            # our data range is [-0.5,0.5], Madry's model is [0,1]
            self.x_input = data + 0.5
            self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

            # first convolutional layer
            W_conv1 = self._weight_variable([5,5,1,32])
            b_conv1 = self._bias_variable([32])

            h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            # second convolutional layer
            W_conv2 = self._weight_variable([5,5,32,64])
            b_conv2 = self._bias_variable([64])

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            # first fully connected layer
            W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self._bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # output layer
            W_fc2 = self._weight_variable([1024,10])
            b_fc2 = self._bias_variable([10])

            pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

            if self.use_log:
                output = tf.nn.softmax(pre_softmax)
            else:
                output = pre_softmax
        
            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]

            # remove the scope name during reload
            var_trans_dict = {}
            for var in new_vars:
                var_trans_dict[var.op.name.replace(name_prefix + '/', '')] = var

            # restore model
            saver = tf.train.Saver(var_list=var_trans_dict)
            saver.restore(self.sess, self.model_file)
            # self.model.output = output
            # self.model.input = data

        return output

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
              ksize = [1,2,2,1],
              strides=[1,2,2,1],
              padding='SAME')

