import numpy as np
import json
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow.keras.backend as K

import datasets
from models.test_model import get_normalize_layer
from models.zoo import Flatten


def load_keras_model(input_shape, path):
    try:
        print(f'Loading keras model {path}')
        # first try keras
        import keras as keras
        model = keras.models.load_model(path, custom_objects={"fn": lambda y_true, y_pred: y_pred, "tf": tf})
    except:
        print(f'Loading tf.keras model {path}')
        # then try tf.keras
        import tf.keras as keras
        model = tf.keras.models.load_model(path, custom_objects={"fn": lambda y_true, y_pred: y_pred, "tf": tf})

    modules = list()

    # model.summary()
    first_w = True

    for layer in model.layers:
        if isinstance(layer, keras.layers.core.Flatten):
            # print(layer)
            modules.append(Flatten())

        elif isinstance(layer, keras.layers.core.Dense):
            # print(layer)
            # print(layer.activation)
            # print(layer.use_bias)
            # print(layer.kernel)
            # print(layer.bias)

            linear = nn.Linear(layer.input_shape[1], layer.output_shape[1])
            w, b = layer.get_weights()
            if not first_w:
                linear.weight.data.copy_(torch.Tensor(w.T.copy()))
            else:
                permutation = list()
                # permute the last channel to the first
                c, hh, ww = input_shape
                for i in range(c):
                    for j in range(hh):
                        for k in range(ww):
                            permutation.append(j * ww * c + k * c + i)
                old_weight = w.T.copy()
                new_weight = old_weight[:, permutation]
                linear.weight.data.copy_(torch.Tensor(new_weight))
                first_w = False

            linear.bias.data.copy_(torch.Tensor(b))
            modules.append(linear)

        elif isinstance(layer, keras.layers.core.Activation):
            # print(layer)
            # print(layer.activation)

            if 'relu' in str(layer.activation):
                modules.append(nn.ReLU())
            elif 'tanh' in str(layer.activation):
                modules.append(nn.Tanh())
            else:
                raise (ValueError("Unsupported activation"))

        elif isinstance(layer, keras.layers.advanced_activations.LeakyReLU):
            # print(layer)
            # print(layer.alpha)

            modules.append(nn.LeakyReLU(layer.alpha))

        elif isinstance(layer, keras.layers.core.Dropout):

            modules.append(nn.Dropout(layer.rate))
        else:
            raise Exception('Unsupported layer', type(layer))

    ret = nn.Sequential(*modules)
    # print(ret)
    ret = ret.cuda()
    return ret


def abstract_load_keras_model(folder, dataset, num_layer, activation, hidden_neurons, mode=None):
    """
    Parse parameters to path in a friendly way
    :param folder: 'fastlin' or 'recurjac'
    :param dataset: 'cifar' or 'mnist'
    :param num_layer: positive integer
    :param activation: 'relu' or 'leaky' or 'tanh'
    :param hidden_neurons: positive integer
    :param mode: None or 'best' or 'adv_retrain' or 'distill'
    :return: the pytorch model
    """
    assert folder in ['fastlin', 'recurjac']
    assert dataset in ['cifar', 'mnist']
    assert type(num_layer) == int and num_layer > 0
    assert activation in ['relu', 'leaky', 'tanh']
    assert type(hidden_neurons) == int and hidden_neurons > 0
    assert mode is None or mode in ['best', 'adv_retrain', 'distill']
    path_str = f'models_weights/models_{folder}/{dataset}_{num_layer}layer_{activation}_{hidden_neurons}'

    input_shape = {'cifar': 'cifar10', 'mnist': 'mnist'}[dataset]
    input_shape = datasets.get_input_shape(input_shape)

    if mode is not None:
        path_str += f'_{mode}'
    return load_keras_model(input_shape, path_str)


# In Recurjac models, the image channels are the first...
# K.set_image_data_format('channels_first')
