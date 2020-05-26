import torch
import torch.nn as nn
import numpy as np

from recurjac.mnist_cifar_models import NLayerModel
from recurjac.bound_base import get_weights_list, compute_bounds, compute_bounds_integral
from recurjac.bound_spectral import spectral_bound
from recurjac.utils import binary_search

from adaptor.cnncert_adaptor import check_consistency

import datasets
from datasets import NormalizeLayer
from basic.models import FlattenConv2D, model_transform
from models.test_model import Flatten
from adaptor.basic_adaptor import VerifierAdaptor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

global graph
graph = tf.get_default_graph()
global sess
sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))


def torch2keras(dataset, model):
    with sess.as_default():
        with graph.as_default():
            input_shape = datasets.get_input_shape(dataset)
            ans = keras.Sequential()
            n = 0
            activation, activation_param = list(), None
            first_layer = True

            for layer in model:

                if first_layer:
                    kwargs = {'input_shape': input_shape}
                    first_layer = False
                else:
                    kwargs = {}

                n += 1
                if isinstance(layer, Flatten):
                    ans.add(keras.layers.Flatten('channels_last', **kwargs))
                elif isinstance(layer, nn.Linear) or isinstance(layer, FlattenConv2D):
                    i, o = layer.in_features, layer.out_features
                    l = keras.layers.Dense(o)
                    ans.add(l)
                    l.set_weights([layer.weight.t().cpu().detach().numpy(), layer.bias.cpu().detach().numpy()])
                elif isinstance(layer, nn.ReLU):
                    ans.add(keras.layers.Activation('relu', name=f'relu_{n}'))
                    activation.append('relu')
                elif isinstance(layer, nn.Tanh):
                    ans.add(keras.layers.Activation('tanh', name=f'tanh_{n}'))
                    activation.append('tanh')
                elif isinstance(layer, nn.LeakyReLU):
                    ans.add(keras.layers.LeakyReLU(alpha=layer.negative_slope, name=f'leaky_{n}'))
                    activation.append('leaky')
                    activation_param = layer.negative_slope
                elif isinstance(layer, nn.Dropout):
                    # ignore dropout layer since we only use the model for evaluation here
                    pass
                elif isinstance(layer, nn.Conv2d):
                    new_layer = keras.layers.Conv2D(layer.out_channels, layer.kernel_size, layer.stride,
                                                    'valid' if layer.padding[0] == 0 else 'same',
                                                    'channels_first',
                                                    use_bias=layer.bias is not None,
                                                    **kwargs)

                    ans.add(new_layer)
                    # print(ret.output_shape)
                    new_weights = [layer.weight.cpu().detach().numpy().transpose(2, 3, 1, 0)]
                    if layer.bias is not None:
                        new_weights.append(layer.bias.cpu().detach().numpy())
                    new_layer.set_weights(new_weights)
                else:

                    raise NotImplementedError

    # only one type of activation is permitted
    activation = list(set(activation))
    assert len(activation) == 1
    activation = activation[0]

    return ans, activation, activation_param


class RecurBaseModel(NLayerModel):
    """
        The class functioning like NLayerModel in the original framework.
    """
    def __init__(self, dataset, model):
        self.model, self.activation, self.activation_param = torch2keras(dataset,
                                                                         model_transform(model, datasets.get_input_shape(dataset)))

        print(self.model.summary())

        with sess.as_default():
            with graph.as_default():
                try:
                    assert check_consistency(model, self.model, datasets.get_input_shape(dataset), 'channels_first') == True
                except:
                    raise Exception("Somehow the transformed model behaves differently from the original model.")

        with sess.as_default():
            with graph.as_default():
                # extract weights
                self.U = list()
                for layer in self.model.layers:
                    if isinstance(layer, keras.layers.Dense):
                        self.U.append(layer)

                self.W = self.U[-1]
                self.U = self.U[:-1]

                layer_outputs = []
                # save the output of intermediate layers
                for layer in self.model.layers:
                    if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                        layer_outputs.append(K.function([self.model.layers[0].input], [layer.output]))

                # a tensor to get gradients
                self.gradients = []
                for i in range(self.model.output.shape[1]):
                    output_tensor = self.model.output[:, i]
                    self.gradients.append(K.gradients(output_tensor, self.model.input)[0])

                self.layer_outputs = layer_outputs
                self.model.summary()


class RecurJacBase(VerifierAdaptor):

    def __init__(self, dataset, model):
        super(RecurJacBase, self).__init__(dataset, model)

        self.model = RecurBaseModel(dataset, self.model)

        with sess.as_default():
            with graph.as_default():
                # the weights and bias are saved in lists: weights and bias
                # weights[i-1] gives the ith layer of weight and so on
                self.weights, self.biases = get_weights_list(self.model)

        # hyperparameters
        self.lipsteps = 15
        self.layerbndalg = 'crown-adaptive' if self.model.activation == 'relu' else 'crown-general'
        self.bounded_input = True
        self.jacbndalg = None
        # to be concretized to 'fastlip' or 'recurjac'
        self.lipsdir = -1
        self.lipsshift = 1
        self.steps = 15

    def verify(self, input, label, norm_type, radius) -> bool:
        norm = {'1': 1, '2': 2, 'inf': np.inf}[norm_type]
        input = self.input_preprocess(input)
        with sess.as_default():
            with graph.as_default():
                preds = self.model.model.predict(input.unsqueeze(0).numpy())
        pred = preds[0]
        pred_label = np.argmax(pred, axis=0)
        if pred_label != label:
            return 0.0
        else:
            m_radius = radius / self.coef
            input = input.numpy()
            if self.jacbndalg != 'disable':
                robustness_lb = compute_bounds_integral(
                    self.weights, self.biases, pred_label, -1, input,
                    pred, len(self.weights), norm, m_radius, self.lipsteps,
                    self.layerbndalg, self.jacbndalg, untargeted=True,
                    activation=self.model.activation,
                    activation_param=self.model.activation_param, lipsdir=self.lipsdir,
                    lipsshift=self.lipsshift)
                return robustness_lb == m_radius
            else:
                gap_gx, _, _, _ = compute_bounds(self.weights, self.biases, pred_label, -1, input, pred,
                                                 len(self.weights), norm, m_radius, self.layerbndalg, "disable",
                                                 untargeted=True, use_quad=False,
                                                 activation=self.model.activation,
                                                 activation_param=self.model.activation_param,
                                                 bounded_input=True)
                return gap_gx >= 0

    def calc_radius(self, input, label, norm_type, upper=0.5, eps=1e-4) -> float:
        norm = {'1': 1, '2': 2, 'inf': np.inf}[norm_type]
        input = self.input_preprocess(input)
        with sess.as_default():
            with graph.as_default():
                preds = self.model.model.predict(input.unsqueeze(0).numpy())
        pred = preds[0]
        pred_label = np.argmax(pred, axis=0)
        if pred_label != label:
            return 0.0
        else:
            input = input.numpy()
            if self.jacbndalg != 'disable':
                def binary_search_cond(current_eps):
                    robustness_lb = compute_bounds_integral(
                        self.weights, self.biases, pred_label, -1, input,
                        pred, len(self.weights), norm, current_eps, self.lipsteps,
                        self.layerbndalg, self.jacbndalg, untargeted=True,
                        activation=self.model.activation,
                        activation_param=self.model.activation_param, lipsdir=self.lipsdir,
                        lipsshift=self.lipsshift)
                    return robustness_lb == current_eps, robustness_lb

                # Using local Lipschitz constant to verify robustness.
                # perform binary search to adaptively find a good eps
                robustness_lb = binary_search(binary_search_cond, upper, max_steps=self.steps)
            else:
                # use linear outer bounds to verify robustness
                def binary_search_cond(current):
                    gap_gx, _, _, _ = compute_bounds(self.weights, self.biases, pred_label, -1, input, pred,
                                                     len(self.weights), norm, current, self.layerbndalg, "disable",
                                                     untargeted=True, use_quad=False,
                                                     activation=self.model.activation, activation_param=self.model.activation_param,
                                                     bounded_input=True)
                    return gap_gx >= 0, gap_gx

                # perform binary search
                robustness_lb = binary_search(binary_search_cond, eps, max_steps=self.steps)

            robustness_lb *= self.coef
            return robustness_lb


class FastLipAdaptor(RecurJacBase):

    def __init__(self, dataset, model):
        super(FastLipAdaptor, self).__init__(dataset, model)
        self.jacbndalg = 'fastlip'


class RecurJacAdaptor(RecurJacBase):

    def __init__(self, dataset, model):
        super(RecurJacAdaptor, self).__init__(dataset, model)
        self.jacbndalg = 'recurjac'


class SpectralAdaptor(RecurJacBase):

    def __init__(self, dataset, model):
        super(SpectralAdaptor, self).__init__(dataset, model)
        self.layerbndalg = 'spectral'

    def verify(self, input, label, norm_type, radius):
        return radius <= self.calc_radius(input, label, norm_type)

    def calc_radius(self, input, label, norm_type, upper=0.5, eps=1e-4) -> float:
        norm = {'1': 1, '2': 2, 'inf': np.inf}[norm_type]
        with sess.as_default():
            with graph.as_default():
                preds = self.model.model.predict(input.unsqueeze(0).numpy())
        pred = preds[0]
        pred_label = np.argmax(pred, axis=0)
        if pred_label != label:
            return 0.0
        else:
            input = input.numpy()
            robustness_lb, _ = spectral_bound(self.weights, self.biases, pred_label, -1, input, pred,
                                              len(self.weights), self.model.activation, norm, True)
            robustness_lb *= self.coef
            return robustness_lb


class FastLinAdaptor(RecurJacBase):

    def __init__(self, dataset, model):
        super(FastLinAdaptor, self).__init__(dataset, model)
        self.jacbndalg = 'disable'
        self.layerbndalg = 'fastlin'



