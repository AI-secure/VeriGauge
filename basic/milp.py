
import torch
from torch import nn
from models.zoo import Flatten
from basic.models import FlattenConv2D

import numpy as np
import cvxpy as cp


class MILPVerifier:

    def __init__(self, model, in_shape, in_min, in_max):
        super(MILPVerifier, self).__init__()

        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()

        i = 0
        for l in model:
            if i == 0:
                assert isinstance(l, Flatten)
                in_numel = in_shape[0] * in_shape[1] * in_shape[2]
                shapes.append(in_numel)
            else:
                if i % 2 == 1:
                    if isinstance(l, FlattenConv2D):
                        assert shapes[-1] == l.in_numel
                        now_shape = l.out_numel
                        Ws.append(l.weight.detach().cpu().numpy())
                        bs.append(l.bias.detach().cpu().numpy())
                    elif isinstance(l, nn.Linear):
                        assert shapes[-1] == l.in_features
                        now_shape = l.out_features
                        Ws.append(l.weight.detach().cpu().numpy())
                        bs.append(l.bias.detach().cpu().numpy())
                    else:
                        raise Exception("Unexpected layer type")
                    shapes.append(now_shape)
                else:
                    assert isinstance(l, nn.ReLU)
            i = i + 1
        assert num_layers % 2 == 0

        self.in_min, self.in_max = in_min, in_max

        self.in_numel = in_numel
        self.num_layers = num_layers
        self.shapes = shapes
        self.Ws = Ws
        self.bs = bs

    def construct(self, l, u, x0, eps):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()
        self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)
        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        pre = self.cx
        for i in range(len(self.Ws) - 1):
            now_x = (self.Ws[i] @ pre) + self.bs[i]
            now_shape = self.shapes[i + 1]
            now_y = cp.Variable(now_shape)
            now_a = cp.Variable(now_shape, boolean=True)
            for j in range(now_shape):
                if l[i + 1][j] >= 0:
                    self.constraints.extend([now_y[j] == now_x[j]])
                elif u[i + 1][j] <= 0:
                    self.constraints.extend([now_y[j] == 0.])
                else:
                    self.constraints.extend([
                        (now_y[j] <= now_x[j] - (1 - now_a[j]) * l[i + 1][j]),
                        (now_y[j] >= now_x[j]),
                        (now_y[j] <= now_a[j] * u[i + 1][j]),
                        (now_y[j] >= 0.)
                    ])
            # self.constraints.extend([(now_y <= now_x - cp.multiply((1 - now_a), l[i + 1])), (now_y >= now_x), (now_y <= cp.multiply(now_a, u[i + 1])), (now_y >= 0)])
            pre = now_y
        self.last_x = pre

    def prepare_verify(self, y0, yp):
        last_w = self.Ws[-1]
        last_b = self.bs[-1][y0] - self.bs[-1][yp]
        obj = cp.Minimize(((last_w[y0] - last_w[yp]) @ self.last_x) + last_b)
        self.prob = cp.Problem(obj, self.constraints)

