
import torch
from torch import nn

import sys
import cvxpy as cp
import numpy as np

from models.zoo import Flatten
from basic.models import FlattenConv2D


class PercySDP():
    """
        Reimplementation of Percy Liang's SDP paper
    """

    def __init__(self, model, in_shape, timeout=50, threads=30):
        super(PercySDP, self).__init__()

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

        self.in_numel = in_numel
        self.num_layers = num_layers
        self.shapes = shapes
        self.tot_n = sum(self.shapes[:-1]) + 1
        self.Ws = Ws
        self.bs = bs

        self.P = cp.Variable((self.tot_n, self.tot_n), symmetric=True)

        self.timeout = timeout
        self.threads = threads

    def run(self, bl, bu, y0, yp):
        constraints = [self.P >> 0, self.P[0][0] == 1.0]

        p = 1 + self.shapes[0]
        for i in range(1, len(self.shapes) - 1):
            constraints.append(self.P[p: (p + self.shapes[i]), 0] >= 0)
            constraints.append(self.P[p: (p + self.shapes[i]), 0] >= cp.matmul(self.Ws[i-1], self.P[(p - self.shapes[i-1]): p, 0]) + self.bs[i-1])
            constraints.append(cp.diag(self.P)[p: (p + self.shapes[i])] == cp.diag(cp.matmul(self.Ws[i-1], self.P[(p - self.shapes[i-1]): p, p: (p + self.shapes[i])])) + cp.multiply(self.bs[i-1], self.P[p: (p + self.shapes[i]), 0]))
            p += self.shapes[i]

        p = 1
        for i in range(0, len(self.shapes) - 1):
            constraints.append(cp.diag(self.P)[p: (p + self.shapes[i])] <= cp.multiply((bl[i] + bu[i]), self.P[p: (p + self.shapes[i]), 0]) - np.multiply(bl[i], bu[i]))
            p += self.shapes[i]

        obj = (self.Ws[-1][yp] - self.Ws[-1][y0]) * self.P[- self.shapes[-2]:, 0] + self.bs[-1][yp] - self.bs[-1][y0]

        self.prob = cp.Problem(cp.Maximize(obj), constraints)
        # self.prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={
        #     # 'optimizerMaxTime': self.timeout,
        #     'MSK_DPAR_OPTIMIZER_MAX_TIME': self.timeout,
        #     # 'numThreads': self.threads,
        #     'MSK_IPAR_NUM_THREADS': self.threads,
        #     # 'lowerObjCut': 0.,
        #     'MSK_DPAR_LOWER_OBJ_CUT': 0.,
        # })
        self.prob.solve(solver=cp.SCS, verbose=True, warm_start=True, eps=1e-2)
        print('status:', self.prob.status)
        print('optimal value:', self.prob.value)



