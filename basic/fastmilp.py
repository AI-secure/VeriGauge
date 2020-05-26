import gurobipy as gp
from gurobipy import GRB

import torch
from torch import nn
from models.zoo import Flatten

import numpy as np
import time

BIGFLOAT = 1e+20


class FastMILPVerifier:

    def __init__(self, model, in_shape, in_min, in_max, timeout=60, threads=30):
        super(FastMILPVerifier, self).__init__()
        self.model = model
        self.in_shape, self.in_min, self.in_max = in_shape, in_min, in_max
        self.timeout = timeout
        self.threads = threads

    def construct(self, l, u, x0, eps):

        cur_shape = self.in_shape
        interval_ptr = 0

        m = gp.Model()
        # print(self.in_min, self.in_max)
        x_in = m.addVars(*self.in_shape, lb=self.in_min, ub=self.in_max, vtype=GRB.CONTINUOUS, name="x_in")
        # print(eps)
        low_x, high_x = x0 - eps, x0 + eps
        low_x = low_x.clamp(min=self.in_min, max=self.in_max)
        high_x = high_x.clamp(min=self.in_min, max=self.in_max)
        m.addConstrs((x_in[i,j,k] >= low_x[i][j][k] for i in range(low_x.shape[0])
                      for j in range(low_x.shape[1]) for k in range(low_x.shape[2])), "x_in_lower")
        m.addConstrs((x_in[i,j,k] <= high_x[i][j][k] for i in range(high_x.shape[0])
                      for j in range(high_x.shape[1]) for k in range(high_x.shape[2])), "x_in_upper")

        vars = [x_in]

        for no, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                print(f'linear_{no}')
                interval_ptr += 1
                out_n, in_n = layer.weight.shape[0], layer.weight.shape[1]
                weight = layer.weight.detach().cpu().numpy().tolist()
                bias = layer.bias.detach().cpu().numpy().tolist()
                x = vars[-1]
                y = m.addVars(out_n, lb=-BIGFLOAT, vtype=GRB.CONTINUOUS, name=f'x_{no}')
                m.addConstrs((((gp.quicksum((x[jj] * weight[ii][jj]) for jj in range(in_n)))
                               + bias[ii] == y[ii]) for ii in range(out_n)), f"linear_{no}")
                vars.append(y)
                cur_shape = (out_n,)
            elif isinstance(layer, nn.Conv2d):
                print(f'conv_{no}')

                interval_ptr += 1

                assert layer.dilation == (1, 1)
                assert layer.groups == 1
                out_shape = (layer.out_channels,
                             (cur_shape[1] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1,
                             (cur_shape[2] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1)
                x = vars[-1]
                y = m.addVars(*out_shape, lb=-BIGFLOAT, vtype=GRB.CONTINUOUS, name=f'x_{no}')

                conv_weight = layer.weight.tolist()
                conv_bias = layer.bias.tolist()
                padding = layer.padding
                stride = layer.stride
                kernel_size = layer.kernel_size
                # weight_shape = tuple(conv_weight.size())

                m.addConstrs((
                    (gp.quicksum(
                        conv_weight[o][i][jj][kk] * x[i, (-padding[0] + stride[0] * j + jj), (-padding[1] + stride[1] * k + kk)]
                        for jj in range(kernel_size[0])
                        for kk in range(kernel_size[1])
                        # for jj in range(max(0, -(-padding[0] + stride[0] * j)),
                        #                     min(kernel_size[0], cur_shape[1] - (-padding[0] + stride[0] * j)))
                        # for kk in range(max(0, -(-padding[1] + stride[1] * k)),
                        #                     min(kernel_size[1], cur_shape[2] - (-padding[1] + stride[1] * k)))
                        for i in range(cur_shape[0])

                        if 0 <= - padding[0] + stride[0] * j + jj < cur_shape[1]
                           and 0 <= - padding[1] + stride[1] * k + kk < cur_shape[2]
                    ) + conv_bias[o] == y[o, j, k])
                    for o in range(out_shape[0]) for j in range(out_shape[1]) for k in range(out_shape[2])),
                    f"conv_{no}")

                vars.append(y)
                l[interval_ptr] = np.reshape(l[interval_ptr], out_shape)
                u[interval_ptr] = np.reshape(u[interval_ptr], out_shape)
                cur_shape = out_shape

            elif isinstance(layer, nn.ReLU):
                print(f'relu_{no}')

                x = vars[-1]
                y = m.addVars(*cur_shape, vtype=GRB.CONTINUOUS, name=f'x_{no}')
                a = m.addVars(*cur_shape, vtype=GRB.BINARY, name=f'a_{no}')
                if len(cur_shape) == 1:
                    m.addConstrs(((y[i] <= x[i] - (1.0 - a[i]) * l[interval_ptr][i])
                                 for i in range(cur_shape[0]) if l[interval_ptr][i] < 0. < u[interval_ptr][i]),
                                 name='relu_cons_1')
                    m.addConstrs(((y[i] >= x[i])
                                 for i in range(cur_shape[0]) if l[interval_ptr][i] < 0. < u[interval_ptr][i]),
                                 name='relu_cons_2')
                    m.addConstrs(((y[i] <= a[i] * u[interval_ptr][i])
                                 for i in range(cur_shape[0]) if l[interval_ptr][i] < 0. < u[interval_ptr][i]),
                                 name='relu_cons_3')
                    m.addConstrs(((y[i] >= 0)
                                 for i in range(cur_shape[0]) if l[interval_ptr][i] < 0. < u[interval_ptr][i]),
                                 name='relu_cons_4')
                    m.addConstrs(((y[i] == x[i]) for i in range(cur_shape[0]) if l[interval_ptr][i] >= 0), name='relu_ident')
                    m.addConstrs(((a[i] == 1) for i in range(cur_shape[0]) if l[interval_ptr][i] >= 0), name='relu_ident')
                    m.addConstrs(((y[i] == 0) for i in range(cur_shape[0]) if u[interval_ptr][i] <= 0), name='relu_zero')
                    m.addConstrs(((a[i] == 0) for i in range(cur_shape[0]) if u[interval_ptr][i] <= 0), name='relu_zero')

                elif len(cur_shape) == 2:
                    m.addConstrs(((y[i, j] <= x[i, j] - l[interval_ptr][i][j] * (1.0 - a[i,j]))
                                 for i in range(cur_shape[0]) for j in range(cur_shape[1])
                                 if l[interval_ptr][i][j] < 0. < u[interval_ptr][i][j]), name='relu_cons_1')
                    m.addConstrs(((y[i, j] >= x[i, j])
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1])
                                  if l[interval_ptr][i][j] < 0. < u[interval_ptr][i][j]), name='relu_cons_2')
                    m.addConstrs(((y[i, j] <= u[interval_ptr][i][j] * a[i, j])
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1])
                                  if l[interval_ptr][i][j] < 0. < u[interval_ptr][i][j]), name='relu_cons_3')
                    m.addConstrs(((y[i, j] >= 0)
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1])
                                  if l[interval_ptr][i][j] < 0. < u[interval_ptr][i][j]), name='relu_cons_4')
                    m.addConstrs(((y[i][j] == x[i][j]) for i in range(cur_shape[0]) for j in range(cur_shape[1]) if l[interval_ptr][i][j] >= 0), name='relu_ident')
                    m.addConstrs(((y[i][j] == 0) for i in range(cur_shape[0]) for j in range(cur_shape[1]) if u[interval_ptr][i][j] <= 0), name='relu_zero')

                elif len(cur_shape) == 3:
                    m.addConstrs(((y[i, j, k] <= x[i, j, k] - l[interval_ptr][i][j][k] * (1.0 - a[i, j, k]))
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])
                                  if l[interval_ptr][i][j][k] < 0. < u[interval_ptr][i][j][k]), name='relu_cons_1')
                    m.addConstrs(((y[i, j, k] >= x[i, j, k])
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])
                                  if l[interval_ptr][i][j][k] < 0. < u[interval_ptr][i][j][k]), name='relu_cons_2')
                    m.addConstrs(((y[i, j, k] <= u[interval_ptr][i][j][k] * a[i, j, k])
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])
                                  if l[interval_ptr][i][j][k] < 0. < u[interval_ptr][i][j][k]), name='relu_cons_3')
                    m.addConstrs(((y[i, j, k] >= 0)
                                  for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])
                                  if l[interval_ptr][i][j][k] < 0. < u[interval_ptr][i][j][k]), name='relu_cons_4')
                    m.addConstrs(
                        ((y[i, j, k] == x[i, j, k])
                         for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])
                         if l[interval_ptr][i][j][k] >= 0),
                        name='relu_ident'
                    )
                    m.addConstrs(
                        ((y[i, j, k] == 0)
                         for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])
                         if u[interval_ptr][i][j][k] <= 0),
                        name='relu_zero'
                    )
                else:
                    raise Exception('Unsupported dimension')
                vars.append(a)
                vars.append(y)
                assert tuple(l[interval_ptr].shape) == cur_shape
            elif isinstance(layer, Flatten):
                print(f'flatten_{no}')
                tmp = 1
                for j in cur_shape:
                    tmp *= j

                x = vars[-1]
                y = m.addVars(tmp, lb=-BIGFLOAT, vtype=GRB.CONTINUOUS, name=f'x_{no}')

                if len(cur_shape) == 1:
                    m.addConstrs(x == y, name=f'flatten_{no}')
                elif len(cur_shape) == 2:
                    m.addConstrs(((y[i * cur_shape[1] + j] == x[i, j]) for i in range(cur_shape[0]) for j in range(cur_shape[1])), name=f'flatten_{no}')
                elif len(cur_shape) == 3:
                    m.addConstrs(((y[i * cur_shape[1] * cur_shape[2] + j * cur_shape[2] + k] == x[i, j, k])
                                 for i in range(cur_shape[0]) for j in range(cur_shape[1]) for k in range(cur_shape[2])), name=f'flatten_{no}')
                vars.append(y)

                cur_shape = (tmp,)
            else:
                raise NotImplementedError()

        self.m_out = vars[-1]
        self.m = m

    def verify(self, y0, yp):

        self.m.setObjective(self.m_out[y0] - self.m_out[yp], GRB.MINIMIZE)

        self.m.setParam('LogToConsole', 0)
        self.m.setParam("TimeLimit", self.timeout)
        self.m.setParam('BestObjStop', -1e-6)
        self.m.setParam('BestBdStop', +1e-6)
        self.m.setParam('InfUnbdInfo', 1)

        if self.threads is not None:
            self.m.setParam('Threads', self.threads)

        self.m.optimize()

        if self.m.status == GRB.OPTIMAL or self.m.status == GRB.USER_OBJ_LIMIT:
            if self.m.ObjVal < 0:
                return False
            elif self.m.ObjBound > 0:
                return True
        else:
            return False


