import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import torch
from torch import nn

from cnn_cert.cnn_bounds_full import loss, ResidualStart, ResidualStart2
from adaptor.cnncert_adaptor import check_consistency
import models.zoo


def load_from_tf(path):
    keras_model = load_model(path,
                             custom_objects={'fn':loss, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2,
                                             'tf':tf})
    return keras_model
    # print(path)
    # print(keras_model.summary())


def sequential_keras2torch(keras_model, default_activation='relu'):
    assert isinstance(keras_model, keras.models.Sequential)
    model = list()
    transposed = False
    last_shape = keras_model.input_shape[1:]

    for layer in keras_model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            kernel_size = layer.kernel_size
            strides = layer.strides
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            last_shape = output_shape[1:]

            pad_h = (output_shape[1] - 1) * strides[0] + kernel_size[0] - input_shape[1]
            pad_w = (output_shape[2] - 1) * strides[1] + kernel_size[1] - input_shape[2]
            assert pad_h % 2 == 0 and pad_w % 2 == 0

            weights = layer.get_weights()
            if len(weights) == 1:
                use_bias = False
            else:
                use_bias = True

            assert not transposed
            weights[0] = weights[0].transpose((3, 2, 0, 1))

            now_layer = nn.Conv2d(in_channels=input_shape[-1], out_channels=output_shape[-1], kernel_size=kernel_size,
                                  stride=strides, padding=(pad_h // 2, pad_w // 2), bias=use_bias)

            now_layer.weight = nn.Parameter(torch.tensor(weights[0]))
            if use_bias:
                now_layer.bias = nn.Parameter(torch.tensor(weights[1]))
            model.append(now_layer)

        elif isinstance(layer, keras.layers.Activation) or isinstance(layer, keras.layers.Lambda):
            if isinstance(layer, keras.layers.Lambda):
                name = default_activation
            else:
                name = layer.activation.__name__
            if name == 'relu':
                model.append(nn.ReLU())
            elif name == 'tanh':
                model.append(nn.Tanh())
            elif name == 'sigmoid':
                model.append(nn.Sigmoid())
            else:
                raise Exception(f'Unsupported activation {name}')

        elif isinstance(layer, keras.layers.MaxPool2D):
            kernel_size = layer.pool_size
            strides = layer.strides
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            last_shape = output_shape[1:]

            pad_h = (output_shape[1] - 1) * strides[0] + kernel_size[0] - input_shape[1]
            pad_w = (output_shape[2] - 1) * strides[1] + kernel_size[1] - input_shape[2]
            assert pad_h % 2 == 0 and pad_w % 2 == 0

            now_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=strides, padding=(pad_h // 2, pad_w // 2))
            model.append(now_layer)

        elif isinstance(layer, keras.layers.Flatten):
            model.append(models.zoo.Flatten())

        elif isinstance(layer, keras.layers.Dropout):
            rate = layer.rate
            model.append(nn.Dropout(p=rate))

        elif isinstance(layer, keras.layers.BatchNormalization):
            if len(layer.input_shape) == 4:
                # 2D
                now_layer = nn.BatchNorm2d(num_features=layer.input_shape[1])
                weight, bias = layer.get_weights()
                now_layer.weight = nn.Parameter(torch.Tensor(weight))
                now_layer.bias = nn.Parameter(torch.Tensor(bias))
                model.append(now_layer)
            else:
                raise Exception(f'Unsupported batch norm dim {len(layer.input_shape) - 1}')

        elif isinstance(layer, keras.layers.Dense):

            in_features = layer.input_shape[1]
            out_features = layer.output_shape[1]
            weight = layer.get_weights()
            if len(weight) == 1:
                bias = None
            else:
                weight, bias = weight
            if not transposed:
                permutation = [j * last_shape[1] * last_shape[2] + k * last_shape[2] + i for i in range(last_shape[2]) for j in range(last_shape[0]) for k in range(last_shape[1])]
                weight = weight[permutation]
                transposed = True
            weight = weight.T

            now_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias is not None)
            now_layer.weight = nn.Parameter(torch.Tensor(weight))
            if bias is not None:
                now_layer.bias = nn.Parameter(torch.Tensor(bias))
            model.append(now_layer)

        else:
            raise Exception(f'Unsupported layer type {layer.__class__.__name__}')

    return nn.Sequential(*model)


model_root_path = 'models_weights/cnn_cert_models'


def load_cnn_cert_model(fname):
    print(f'Load model {fname}')
    # the real load function
    m = load_from_tf(os.path.join(model_root_path, fname))

    possible_act = fname.split('_')[-1]
    if possible_act in ['sigmoid', 'tanh']:
        default_act = possible_act
    else:
        default_act = 'relu'

    m.summary()
    p_m = sequential_keras2torch(m, default_act)
    # print(p_m)
    p_m = p_m.cuda()
    return p_m

# ======================
# The following script identifies which models are really usable in our environment
# We find out the following models are available in our environment, listed in `mnist_models' and `cifar_models'

# success_list = list()
# failed_list = list()
#
# tot = len(os.listdir(model_root_path))
#
# for i,fname in enumerate(os.listdir(model_root_path)):
#     print(f'{i}/{tot} {fname}')
#     try:
#         model = load_from_tf(os.path.join(model_root_path, fname))
#         success_list.append(fname)
#     except:
#         print(f'Fail to load {fname}')
#         failed_list.append(fname)
#
# print('success', success_list)
# print('failed', failed_list)


mnist_models = [
    'mnist_2layer_fc_20',
    'mnist_3layer_fc_20',
    'mnist_4layer_fc_1024',
    'mnist_cnn_7layer',
    'mnist_cnn_lenet',
    'mnist_cnn_7layer_sigmoid',
    'mnist_cnn_4layer_5_3_sigmoid',
    'mnist_cnn_4layer_5_3_tanh',
    'mnist_cnn_7layer_tanh',
    'mnist_cnn_8layer_5_3_sigmoid',
    'mnist_cnn_8layer_5_3_tanh',
    'mnist_cnn_lenet_sigmoid',
    'mnist_cnn_lenet_tanh',
    # 'mnist_resnet_3_tanh',
    # 'mnist_resnet_3_sigmoid',
    # 'mnist_resnet_4_sigmoid',
    # 'mnist_resnet_5_tanh',
    # 'mnist_resnet_2_sigmoid',
    # 'mnist_resnet_2_tanh',
    # 'mnist_resnet_5_sigmoid',
    # 'mnist_resnet_4_tanh'
]

cifar_models = [
    'cifar_4layer_fc_2048',
    'cifar_5layer_fc_1024',
    'cifar_5layer_fc_2048',
    'cifar_cnn_7layer',
    'cifar_7layer_fc_1024',
    'cifar_cnn_5layer_5_3_tanh',
    'cifar_cnn_7layer_5_3_sigmoid',
    'cifar_cnn_7layer_sigmoid',
    'cifar_cnn_7layer_5_3_tanh',
    'cifar_cnn_7layer_tanh',
    'cifar_cnn_5layer_5_3_sigmoid'
]


# ========================
# The following script checks the transformation correctness of our Keras to PyTorch function
# All above models pass the check, only except ResNet models, which are not supported by our function
#
# for i,now in enumerate(mnist_models + cifar_models):
#     print(f'{i}/{len(mnist_models + cifar_models)} {now}')
#     possible_act = now.split('_')[-1]
#     if possible_act in ['sigmoid', 'tanh']:
#         default_act = possible_act
#     else:
#         default_act = 'relu'
#     # now = 'mnist_cnn_7layer'
#     with tf.device('/cpu:0'):
#         m = load_from_tf(os.path.join(model_root_path, now))
#     # m.summary()
#     p_m = sequential_keras2torch(m, default_act)
#     # print(p_m)
#     p_m = p_m.cuda()
#     _, h, w, c = m.input_shape
#     if check_consistency(p_m, m, (c, h, w)):
#         print(now, 'good')
#     else:
#         print('!!!', now, 'bad')


# CNN-cert only supports the image channels to be the last...
# K.set_image_data_format('channels_last')