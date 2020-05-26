import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys

import numpy as np

import datasets
import model
from models.zoo import Flatten

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import gurobipy as gp
from gurobipy import GRB

torch.random.manual_seed(42)

if __name__ == '__main__':

    m = gp.Model()

    x = m.addVars(2, lb=-1.0, vtype=GRB.CONTINUOUS)
    m.addConstrs((x[i] >= -1.0 for i in range(2)), name='lb')
    m.addConstrs((x[i] <= 2.0 for i in range(2)), name='ub')
    # m.addConstr(x[0] >= 1.0, "lb0")
    # m.addConstr(x[0] <= 2.0, "rb0")
    # m.addConstr(x[1] >= 2.0, "lb1")
    # m.addConstr(x[1] <= 4.0, "rb1")
    y = m.addVars(2, lb=-1.0, vtype=GRB.CONTINUOUS)
    m.addConstr(y[1] == x[0])
    m.addConstr(y[0] == x[1])

    m.setObjective(y[0] - y[1], GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.USER_OBJ_LIMIT:
        print(m.ObjVal)
    else:
        raise Exception("MMP")

    # ds = datasets.get_dataset('mnist', 'test')
    # m = model.load_model('exp', 'mnist', 'B')
    # print(m)
    #
    # X, y = ds[0]
    #
    # interm = X.unsqueeze(0)
    # for l in m:
    #     interm = l(interm)
    #     if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.Linear) or isinstance(l, Flatten):
    #         print(l.__class__.__name__)
    #         print(interm)
    #
    # input_shape = datasets.get_input_shape('mnist')
    # ans = keras.Sequential()
    #
    # first_layer = True
    # n = 0
    #
    # for layer in m:
    #
    #     if first_layer:
    #         kwargs = {'input_shape': input_shape}
    #         first_layer = False
    #     else:
    #         kwargs = {}
    #
    #     n += 1
    #     if isinstance(layer, Flatten):
    #         ans.add(keras.layers.Flatten('channels_last', **kwargs))
    #     elif isinstance(layer, nn.Linear):
    #         i, o = layer.in_features, layer.out_features
    #         l = keras.layers.Dense(o)
    #         ans.add(l)
    #         l.set_weights([layer.weight.t().cpu().detach().numpy(), layer.bias.cpu().detach().numpy()])
    #     elif isinstance(layer, nn.ReLU):
    #         ans.add(keras.layers.Activation('relu', name=f'relu_{n}'))
    #     elif isinstance(layer, nn.Tanh):
    #         ans.add(keras.layers.Activation('tanh', name=f'tanh_{n}'))
    #     elif isinstance(layer, nn.LeakyReLU):
    #         ans.add(keras.layers.LeakyReLU(alpha=layer.negative_slope, name=f'leaky_{n}'))
    #     elif isinstance(layer, nn.Dropout):
    #         # ignore dropout layer since we only use the model for evaluation here
    #         pass
    #     elif isinstance(layer, nn.Conv2d):
    #         new_layer = keras.layers.Conv2D(layer.out_channels, layer.kernel_size, layer.stride,
    #                                         'valid' if layer.padding[0] == 0 else 'same',
    #                                         'channels_first',
    #                                         use_bias=layer.bias is not None,
    #                                         **kwargs)
    #
    #         ans.add(new_layer)
    #         # print(ret.output_shape)
    #         new_weights = [layer.weight.cpu().detach().numpy().transpose(2, 3, 1, 0)]
    #         if layer.bias is not None:
    #             new_weights.append(layer.bias.cpu().detach().numpy())
    #         new_layer.set_weights(new_weights)
    #     else:
    #         raise NotImplementedError
    #
    # print(ans.summary())
    #
    # pred = m(X.unsqueeze(0))
    # pred2 = ans.predict(np.expand_dims(X.numpy(), 0))
    #
    # interkm = np.expand_dims(X.numpy(), 0)
    # for layer in ans.layers:
    #     interkm = layer(interkm)
    #     if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Flatten):
    #         print(layer.__class__.__name__)
    #         print(K.eval(interkm))
    #
    # print(pred)
    # print(pred2)
