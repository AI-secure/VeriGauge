import numpy as np
import torch
from torch import nn

from basic.models import Flatten, FlattenConv2D
from basic.core import *


class BoundCalculator:

    def __init__(self, model, in_shape, in_min, in_max):
        super(BoundCalculator, self).__init__()

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

        self.l = None
        self.u = None

    def verify(self, y_true, y_adv):
        """
            Assert if y_true >= y_adv holds for all
        :param y_true:
        :param y_adv:
        :return: True: y_true >= y_adv always holds, False: y_true >= y_adv MAY not hold
        """
        assert self.l is not None and self.u is not None
        assert len(self.l) == len(self.Ws)
        assert len(self.u) == len(self.bs)
        assert len(self.l) == len(self.u)
        assert len(self.Ws) == len(self.bs)

        l = self.l[-1]
        u = self.u[-1]
        l = np.maximum(l, 0)
        u = np.maximum(u, 0)
        W = self.Ws[-1]
        b = self.bs[-1]
        W_delta = W[y_true] - W[y_adv]
        b_delta = b[y_true] - b[y_adv]
        lb = np.dot(np.clip(W_delta, a_min=0., a_max=None), l) + np.dot(np.clip(W_delta, a_min=None, a_max=0.), u) + b_delta
        # print(l)
        # print(u)
        # print(u-l)
        # print(y_true, y_adv, lb)
        return lb >= 0.

    def calculate_bound(self, x0, eps):
        raise NotImplementedError("Haven't implemented yet.")


class IntervalBound(BoundCalculator):

    def calculate_bound(self, x0, eps):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.l = [np.clip(x0 - eps, a_min=self.in_min, a_max=self.in_max)]
        self.u = [np.clip(x0 + eps, a_min=self.in_min, a_max=self.in_max)]

        for i in range(len(self.Ws) - 1):
            now_l = self.l[-1]
            now_u = self.u[-1]
            if i > 0:
                now_l = np.clip(now_l, a_min=0., a_max=None)
                now_u = np.clip(now_u, a_min=0., a_max=None)
            W, b = self.Ws[i], self.bs[i]
            new_l = np.matmul(np.clip(W, a_min=0., a_max=None), now_l) + np.matmul(np.clip(W, a_min=None, a_max=0.), now_u) + b
            new_u = np.matmul(np.clip(W, a_min=None, a_max=0.), now_l) + np.matmul(np.clip(W, a_min=0., a_max=None), now_u) + b
            self.l.append(new_l)
            self.u.append(new_u)


class FastLinBound(BoundCalculator):

    def _form_diag(self, l, u):
        d = np.zeros(l.shape[0])
        for i in range(d.shape[0]):
            if u[i] >= 1e-6 and l[i] <= -1e-6:
                d[i] = u[i] / (u[i] - l[i])
            elif u[i] <= -1e-6:
                d[i] = 0.
            else:
                d[i] = 1.
        return np.diag(d)

    def calculate_bound(self, x0, eps):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.l = [np.clip(x0 - eps, a_min=self.in_min, a_max=self.in_max)]
        self.u = [np.clip(x0 + eps, a_min=self.in_min, a_max=self.in_max)]

        A0 = self.Ws[0]
        A = list()

        for i in range(len(self.Ws) - 1):
            T = [None for _ in range(i)]
            H = [None for _ in range(i)]
            for k in range(i - 1, -1, -1):
                if k == i - 1:
                    D = self._form_diag(self.l[-1], self.u[-1])
                    A.append(np.matmul(self.Ws[i], D))
                else:
                    A[k] = np.matmul(A[-1], A[k])
                T[k] = np.zeros_like(A[k].T)
                H[k] = np.zeros_like(A[k].T)
                for r in range(self.l[k+1].shape[0]):
                    if self.u[k+1][r] >= 1e-6 and self.l[k+1][r] <= -1e-6:
                        for j in range(A[k].shape[0]):
                            if A[k][j, r] > 0.:
                                T[k][r, j] = self.l[k+1][r]
                            else:
                                H[k][r, j] = self.l[k+1][r]
            if i > 0:
                A0 = np.matmul(A[-1], A0)
            nowl = list()
            nowu = list()
            for j in range(self.Ws[i].shape[0]):
                nu_j = np.dot(A0[j], x0) + self.bs[i][j]
                mu_p_j = mu_n_j = 0.
                for k in range(0, i):
                    mu_p_j -= np.dot(A[k][j], (T[k].T)[j])
                    mu_n_j -= np.dot(A[k][j], (H[k].T)[j])
                    nu_j += np.dot(A[k][j], self.bs[k])
                nowl.append(mu_n_j + nu_j - eps * np.sum(np.abs(A0[j])))
                nowu.append(mu_p_j + nu_j + eps * np.sum(np.abs(A0[j])))
            self.l.append(np.array(nowl))
            self.u.append(np.array(nowu))


class FastIntervalBound:

    def __init__(self, model, in_shape, in_min, in_max):

        self.in_min, self.in_max = in_min, in_max

        self.l = None
        self.u = None

        self.model = model

    def calculate_bound(self, x0, eps):

        x0 = x0.cuda()
        self.l = [torch.clamp(x0 - eps, min=self.in_min, max=self.in_max)]
        self.u = [torch.clamp(x0 + eps, min=self.in_min, max=self.in_max)]

        hasrelu = False
        with torch.no_grad():
            for l in self.model[:-1]:
                now_l = self.l[-1]
                now_u = self.u[-1]
                if isinstance(l, nn.modules.conv.Conv2d) or isinstance(l, nn.modules.linear.Linear):
                    if hasrelu:
                        now_l = torch.clamp(now_l, min=0.)
                        now_u = torch.clamp(now_u, min=0.)
                    weightp = torch.clamp(l.weight, min=0.)
                    weightn = torch.clamp(l.weight, max=0.)
                    l.weight = nn.Parameter(weightp)
                    new_l = l(now_l.unsqueeze(0))
                    new_u = l(now_u.unsqueeze(0))
                    l.weight = nn.Parameter(weightn)
                    new_l += l(now_u.unsqueeze(0))
                    new_u += l(now_l.unsqueeze(0))
                    l.weight = nn.Parameter(torch.zeros_like(weightp))
                    new_l -= l(now_l.unsqueeze(0))
                    new_u -= l(now_u.unsqueeze(0))
                    l.weight = nn.Parameter(weightp + weightn)
                    new_l = new_l[0]
                    new_u = new_u[0]
                    self.l.append(new_l)
                    self.u.append(new_u)
                elif isinstance(l, nn.modules.activation.ReLU):
                    hasrelu = True
                elif isinstance(l, Flatten):
                    self.l[-1] = self.l[-1].contiguous().reshape(-1)
                    self.u[-1] = self.u[-1].contiguous().reshape(-1)
                else:
                    raise Exception(f"Unsupported layer type: {type(l)}")
            self.l = [x.contiguous().view(-1).cpu().numpy() for x in self.l]
            self.u = [x.contiguous().view(-1).cpu().numpy() for x in self.u]

    def verify(self, y_true, y_adv):
        """
            Assert if y_true >= y_adv holds for all
        :param y_true:
        :param y_adv:
        :return: True: y_true >= y_adv always holds, False: y_true >= y_adv MAY not hold
        """
        with torch.no_grad():
            l = self.l[-1]
            u = self.u[-1]
            l = np.clip(l, a_min=0., a_max=None)
            # torch.clamp(l, min=0.)
            u = np.clip(u, a_min=0., a_max=None)
            # torch.clamp(u, min=0.)
            W = self.model[-1].weight
            b = self.model[-1].bias
            W_delta = W[y_true] - W[y_adv]
            b_delta = b[y_true] - b[y_adv]
            l = np.maximum(l, 0)
            u = np.maximum(u, 0)
            W_delta = W_delta.cpu().numpy()
            b_delta = b_delta.cpu().numpy()
            lb = np.dot(np.clip(W_delta, a_min=0., a_max=None), l) + np.dot(np.clip(W_delta, a_min=None, a_max=0.), u) + b_delta
            lb = np.dot(np.clip(W_delta, a_min=0., a_max=None), l) + np.dot(np.clip(W_delta, a_min=None, a_max=0.), u) + b_delta
            # print(l)
            # print(u)
            # print(u-l)
            # print(y_true, y_adv, lb)
            return lb >= 0.


class IntervalFastLinBound(BoundCalculator):

    def __init__(self, model, in_shape, in_min, in_max):
        super(IntervalFastLinBound, self).__init__(model, in_shape, in_min, in_max)

        self.interval_calc = IntervalBound(model, in_shape, in_min, in_max)
        self.fastlin_calc = FastLinBound(model, in_shape, in_min, in_max)

    def calculate_bound(self, x0, eps):
        self.interval_calc.calculate_bound(x0, eps)
        self.fastlin_calc.calculate_bound(x0, eps)

        self.l = list()
        self.u = list()
        for i in range(len(self.interval_calc.l)):
            self.l.append(np.maximum(self.interval_calc.l[i], self.fastlin_calc.l[i]))
            self.u.append(np.minimum(self.interval_calc.u[i], self.fastlin_calc.u[i]))

