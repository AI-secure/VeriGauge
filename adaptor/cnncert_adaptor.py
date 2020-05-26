import logging
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as la
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD

# This package is useless...
# import pytorch2keras

from adaptor.basic_adaptor import VerifierAdaptor
from datasets import get_input_shape, get_num_classes
from basic.models import FlattenConv2D, model_transform
import models.zoo

from cnn_cert.cnn_bounds_full import *
import cnn_cert.fastlin.save_nlayer_weights as nl
from cnn_cert.fastlin.get_bounds_ours import compute_worst_bound

global graph
global sess

def sequential_torch2keras(torch_model, dataset):
    """
        Transform the sequential torch model on CUDA to Keras
    :param torch_model: the Torch model to transform
    :param dataset: the dataset, typically 'MNIST' or 'CIFAR10'
    :return: the transformed Keras model
    """

    global graph
    global sess
    graph = tf.Graph()
    sess = tf.Session(graph=graph,
                      config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

    with sess.as_default():
        with graph.as_default():
            ret = keras.Sequential()

            assert isinstance(torch_model, nn.Sequential)

            input_shape = get_input_shape(dataset)
            new_input_shape = (input_shape[1], input_shape[2], input_shape[0])

            # before meeting the flatten layer, we transform each channel-first layer to the corresponding channel-last one
            # after meeting the flatten layer, we analyze the first layer and transform the weight matrix
            meet_flatten = False
            shape_before_flatten = None
            transposed = False
            first_layer = True

            for layer in torch_model:
                if first_layer:
                    kwargs = {'input_shape': new_input_shape}
                    first_layer = False
                else:
                    kwargs = {}

                if isinstance(layer, nn.Conv2d):
                    # we don't permit conv2d layer after flatten
                    assert meet_flatten is False
                    # by default, we assume zero-padding size is to allow the "same" padding configuration in keras.Conv2D
                    # since cnn-cert only supports keras.Conv2D but not zero padding layers
                    # print('  in', layer.in_channels)
                    # print('  out', layer.out_channels)
                    # print('  stride', layer.stride)
                    # print('  padding', layer.padding)
                    # print('  paddingmode', layer.padding_mode)
                    # print('  kernelsize', layer.kernel_size)
                    # print('  weight shape', layer.weight.size())
                    # if layer.bias is not None:
                    #     print('  bias shape', layer.bias.size())

                    new_layer = keras.layers.Conv2D(layer.out_channels, layer.kernel_size, layer.stride,
                                                    'valid' if layer.padding[0] == 0 else 'same',
                                                    'channels_last',
                                                    use_bias=layer.bias is not None,
                                                    **kwargs)

                    ret.add(new_layer)
                    # print(ret.output_shape)

                    new_weights = [layer.weight.cpu().detach().numpy().transpose(2, 3, 1, 0)]
                    if layer.bias is not None:
                        new_weights.append(layer.bias.cpu().detach().numpy())
                    new_layer.set_weights(new_weights)

                    # print('  new weight/bias len:', len(new_layer.get_weights()))
                    # print('  new weight shape:', new_layer.get_weights()[0].shape)
                    # print('  new bias shape:', new_layer.get_weights()[1].shape)

                elif isinstance(layer, nn.AvgPool2d):
                    # we don't permit avgpool2d layer after flatten
                    assert meet_flatten is False

                    new_layer = keras.layers.AvgPool2D(layer.kernel_size, layer.stride,
                                                       'valid' if layer.padding[0] == 0 else 'same',
                                                       data_format='channels_last',
                                                       **kwargs)
                    ret.add(new_layer)

                elif isinstance(layer, nn.MaxPool2d):
                    # we don't permit maxpool2d layer after flatten
                    assert meet_flatten is False

                    new_layer = keras.layers.MaxPool2D(layer.kernel_size, layer.stride,
                                                       'valid' if layer.padding[0] == 0 else 'same',
                                                       data_format='channels_last', **kwargs)
                    ret.add(new_layer)

                elif isinstance(layer, nn.ReLU):
                    ret.add(keras.layers.Activation('relu', **kwargs))

                elif isinstance(layer, nn.Tanh):
                    ret.add(keras.layers.Activation('tanh', **kwargs))

                elif isinstance(layer, nn.Sigmoid):
                    ret.add(keras.layers.Activation('sigmoid', **kwargs))

                elif isinstance(layer, models.zoo.Flatten):
                    meet_flatten = True
                    transposed = False
                    if 'input_shape' in kwargs:
                        shape_before_flatten = new_input_shape
                    else:
                        shape_before_flatten = ret.output_shape[1:]
                    ret.add(keras.layers.Flatten(data_format='channels_last', **kwargs))

                elif isinstance(layer, nn.Linear) or isinstance(layer, FlattenConv2D):
                    # print('  in dim', layer.in_features)
                    # print('  out dim', layer.out_features)
                    weights = [layer.weight.cpu().detach().numpy().T]
                    if layer.bias is not None:
                        weights.append(layer.bias.cpu().detach().numpy())
                    # print([x.shape for x in weights])

                    new_layer = keras.layers.Dense(layer.out_features, **kwargs)
                    ret.add(new_layer)
                    # print([x.shape for x in new_layer.get_weights()])

                    if meet_flatten and not transposed:
                        # print('transposed here')
                        h, w, c = shape_before_flatten
                        mapping = [k * h * w + i * w + j for i in range(h) for j in range(w) for k in range(c)]
                        # print(mapping)
                        weights[0] = weights[0][mapping]
                        # print([x.shape for x in weights])
                        transposed = True
                    new_layer.set_weights(weights)

                elif isinstance(layer, nn.Dropout):
                    rate = layer.p
                    ret.add(keras.layers.Dropout(rate))

                else:
                    raise Exception(f'Unsupported layer type {layer.__class__.__name__}')

    return ret


def check_consistency(model, k_model, input_shape, mode='channels_last') -> bool:
    """
        Check the inconsistency between the torch model and the transformed Keras model
        By generating random input data
    :param model: torch model on CUDA
    :param k_model: transformed Keras model
    :param input_shape: input shape for the torch model
    :return: consistent or not
    """
    data = np.random.random(input_shape)
    model.eval()
    pred1 = model(torch.Tensor(data).unsqueeze(0).cuda()).cpu().detach().numpy()
    if mode == 'channels_last':
        pred2 = k_model.predict(np.expand_dims(data.transpose((1,2,0)), 0))
    else:
        pred2 = k_model.predict(np.expand_dims(data, 0))

    # print(pred1)
    # print(pred2)
    # i1 = torch.Tensor(data).unsqueeze(0).cuda()
    # i2 = np.expand_dims(data.transpose((1,2,0)), 0)
    # print(i1)
    # print(i2)
    # for l in model:
    #     print(l.__class__.__name__)
    #     i1 = l(i1)
    #     print(i1)
    # for l in k_model.layers:
    #     print(l.__class__.__name__)
    #     i2 = l(i2)
    #     try:
    #         print(l.data_format)
    #     except:
    #         pass
    #     print(K.get_value(i2))

    err = la.norm(pred1 - pred2)
    precision = err / la.norm(pred1)
    print(f'model difference: {precision*100.0:.3f}%')
    return precision < 1E-3


class CNNCertBase(VerifierAdaptor):

    def input_preprocess(self, input):
        input = super(CNNCertBase, self).input_preprocess(input)
        input = input.permute((1, 2, 0))
        return input

    def __init__(self, dataset, model):
        super(CNNCertBase, self).__init__(dataset, model)

        self.num_classes = get_num_classes(dataset)

        self.activation = list()
        for layer in self.model:
            if isinstance(layer, nn.ReLU):
                self.activation.append('ada')
            elif isinstance(layer, nn.Sigmoid):
                self.activation.append('sigmoid')
            elif isinstance(layer, nn.Tanh):
                self.activation.append('tanh')
        # actually there is another activation called arctan,
        # but there is no corresponding one in pytorch so we ignore it
        self.activation = list(set(self.activation))
        assert len(self.activation) == 1
        self.activation = self.activation[0]

        input_shape = get_input_shape(dataset)
        new_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        self.k_model = sequential_torch2keras(self.model, dataset)

        global graph
        global sess
        with sess.as_default():
            with graph.as_default():

                print(self.k_model.summary())
                try:
                    assert check_consistency(self.model, self.k_model, input_shape) == True
                except:
                    raise Exception("Somehow the transformed model behaves differently from the original model.")

                self.new_model = Model(self.k_model, new_input_shape)

        # Set correct linear_bounds function
        self.linear_bounds = None
        if self.activation == 'relu':
            self.linear_bounds = relu_linear_bounds
        elif self.activation == 'ada':
            self.linear_bounds = ada_linear_bounds
        elif self.activation == 'sigmoid':
            self.linear_bounds = sigmoid_linear_bounds
        elif self.activation == 'tanh':
            self.linear_bounds = tanh_linear_bounds
        elif self.activation == 'arctan':
            self.linear_bounds = atan_linear_bounds



class CNNCertAdaptor(CNNCertBase):

    def verify(self, input, label, norm_type, radius):

        # super L1, L2, Linf norm
        assert norm_type in ['inf', '1', '2']
        p_n = {'inf': 105, '1': 1, '2': 2}[norm_type]

        # radius after preprocessing
        m_radius = radius / self.coef

        input = self.input_preprocess(input)

        global graph
        global sess
        with sess.as_default():
            with graph.as_default():
                preds = self.new_model.model.predict(input.unsqueeze(0).numpy())
                pred = preds[0]
                pred_label = np.argmax(pred, axis=0)

                if pred_label != label:
                    return False

                for target_label in range(self.num_classes):
                    if target_label != pred_label:
                        weights = self.new_model.weights[:-1]
                        biases = self.new_model.biases[:-1]
                        shapes = self.new_model.shapes[:-1]
                        W, b, s = self.new_model.weights[-1], self.new_model.biases[-1], self.new_model.shapes[-1]
                        last_weight = (W[pred_label, :, :, :] - W[target_label, :, :, :]).reshape([1] + list(W.shape[1:]))
                        weights.append(last_weight)
                        biases.append(np.asarray([b[pred_label] - b[target_label]]))
                        shapes.append((1, 1, 1))
                        LB, UB = find_output_bounds(weights, biases, shapes, self.new_model.pads, self.new_model.strides, self.new_model.sizes, self.new_model.types, input.numpy(), m_radius, p_n, self.linear_bounds)
                        if LB <= 0:
                            return False

                return True

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


class FastLinSparseAdaptor(CNNCertBase):

    def __init__(self, dataset, model):
        super(CNNCertBase, self).__init__(dataset, model)

        self.num_classes = get_num_classes(dataset)

        input_shape = get_input_shape(dataset)
        new_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        self.k_model = sequential_torch2keras(model_transform(self.model, input_shape), dataset)

        global graph
        global sess
        with sess.as_default():
            with graph.as_default():
                # Save the transformed Keras model to a temporary place so that the tool can read from file
                # The tool can only init model from file...
                sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
                self.k_model.compile(loss=fn,
                              optimizer=sgd,
                              metrics=['accuracy'])
                self.k_model.save('tmp/tmp.pt')

                self.new_model = nl.CNNModel('tmp/tmp.pt', new_input_shape)
                self.weights = self.new_model.weights
                self.biases = self.new_model.biases

                # print(self.new_model.summary())
                try:
                    check_consistency(self.model, self.k_model, input_shape)
                except Exception:
                    raise Exception("Somehow the transformed model behaves differently from the original model.")

        self.LP = False
        self.LPFULL = False
        self.method = "ours"
        self.dual = False

    def verify(self, input, label, norm_type, radius):

        p = {"inf": "i", "1": "1", "2": "2"}[norm_type]
        m_radius = radius / self.coef

        input = self.input_preprocess(input)

        global graph
        global sess
        with sess.as_default():
            with graph.as_default():
                preds = self.new_model.model.predict(input.unsqueeze(0).numpy())
        pred = preds[0]
        pred_label = np.argmax(pred, axis=0)

        if pred_label != label:
            return False

        target_label = -1
        gap_gx, _, _ = compute_worst_bound(self.weights, self.biases, pred_label, target_label, input.numpy(), pred,
                                           len(self.weights), p, m_radius, self.method, "disable", self.LP, self.LPFULL,
                                           True, self.dual)
        # print(gap_gx)
        return gap_gx >= 0


class LPAllAdaptor(FastLinSparseAdaptor):

    def __init__(self, dataset, model):
        super(LPAllAdaptor, self).__init__(dataset, model)
        self.LPFULL = True


# CNN-cert only supports the image channels to be the last...
# K.set_image_data_format('channels_last')
