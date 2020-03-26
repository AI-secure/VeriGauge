import numpy as np
import torch
import torch.nn as nn

from adaptor.adaptor import Adaptor
from datasets import NormalizeLayer, get_num_classes

from crown_ibp.bound_layers import BoundSequential


class CrownAdaptorBase(Adaptor):
    """
        The base adaptor for CROWN framework
        Note that in constructor, we assume the unnormalized data range is 0 ~ 1
    """

    def __init__(self, dataset, model):
        super(CrownAdaptorBase, self).__init__(dataset, model)
        self.model = model
        self.new_model = None
        # if the normalized layer exists,
        # we need to calculate the norm scaling coefficient
        # when the normalization layer is removed
        # real radius with normalization can thus be figured out
        self.coef = 1.0
        if isinstance(self.model[0], NormalizeLayer):
            self.coef = min(self.model[0].orig_sds)
        self.build_new_model()

        flayer = self.model[0]
        in_min, in_max = 0.0, 1.0
        if isinstance(flayer, NormalizeLayer):
            in_min = (in_min - max(flayer.orig_means))
            in_max = (in_max - min(flayer.orig_means))
            if in_min <= 0.0:
                in_min = in_min / min(flayer.orig_sds)
            else:
                in_min = in_min / max(flayer.orig_sds)
            if in_max <= 0.0:
                in_max = in_max / max(flayer.orig_sds)
            else:
                in_max = in_max / min(flayer.orig_sds)
        self.in_min, self.in_max = in_min, in_max

        self.num_class = get_num_classes(dataset)

    def build_new_model(self):
        # the input is functioned as the canopy
        if isinstance(self.model[0], NormalizeLayer) and isinstance(self.model[1], nn.Sequential):
            # the model is concatenated with a normalized layer at the front
            # just like what we did in models/test_model.py
            self.new_model = self.model[1]
        else:
            self.new_model = self.model
        self.new_model = BoundSequential.convert(self.new_model, {'same-slope': False})

    def input_preprocess(self, input):
        flayer = self.model[0]
        (_, height, width) = input.shape
        if isinstance(flayer, NormalizeLayer):
            input = (input - flayer.means.cpu().repeat(height, width, 1).permute(2, 0, 1)) / flayer.sds.cpu().repeat(height, width, 1).permute(2, 0, 1)
        return input

    def compute(self, model, norm, x_U, x_L, eps, C):
        raise NotImplementedError
        return None, None

    def verify(self, input, label, norm_type, radius) -> bool:
        """
            The default implementation assumes the completeness of calc_radius() function
        :param input: input sample
        :param label: correct label
        :param norm_type: norm type
        :param radius: the floating point radius
        :return: bool
        """
        # super L1, L2, Linf norm
        assert norm_type in ['inf', '1', '2']

        # radius after preprocessing
        m_radius = radius / self.coef

        # preprocess data
        X = self.input_preprocess(input)
        X_L = (X - m_radius).clamp(min=self.in_min)
        X_U = (X + m_radius).clamp(max=self.in_max)
        X = X.unsqueeze(0)
        if norm_type == 'inf':
            X_L = X_L.unsqueeze(0)
            X_U = X_U.unsqueeze(0)
            eps = None
        else:
            # use eps to bound
            X_L = X
            X_U = X
            eps = m_radius
        norm = {'inf': np.inf, '1': 1, '2': 2}[norm_type]


        # pregenerate the array for specifications, will be used for scatter
        sa = np.zeros((self.num_class, self.num_class - 1), dtype=np.int32)
        for i in range(sa.shape[0]):
            for j in range(sa.shape[1]):
                if j < i:
                    sa[i][j] = j
                else:
                    sa[i][j] = j + 1
        sa = torch.LongTensor(sa)

        # generate specifications
        c = torch.eye(self.num_class).type(torch.int32)[label].unsqueeze(0) - torch.eye(self.num_class).type_as(X)
        # remove specifications to self
        I = (~(label == torch.arange(self.num_class)))
        c = (c[I].view(1,self.num_class-1,self.num_class))
        # scatter matrix to avoid compute margin to self
        sa_label = sa[label].unsqueeze(0)
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(X.size(0), self.num_class)

        # move to cuda
        X = X.cuda()
        X_L = X_L.cuda()
        X_U = X_U.cuda()
        c = c.cuda()
        sa_label = sa_label.cuda()
        lb_s = lb_s.cuda()

        self.new_model = self.new_model.cuda()
        self.new_model.eval()

        torch.set_grad_enabled(False)
        regular = self.new_model(X, method_opt="forward", disable_multi_gpu=True)
        regular_pred = regular.argmax(dim=1)[0].item()
        if regular_pred != label:
            return False

        ub, lb = self.compute(self.new_model, norm, X_U, X_L, eps, c)
        torch.set_grad_enabled(True)

        lb = lb_s.scatter(1, sa_label, lb)
        ans = torch.sum((lb < 0).any(dim=1)).cpu().detach().numpy()

        # in fact ans could only be 0 or 1
        return ans < 0.5


class FullCrownAdaptor(CrownAdaptorBase):

    def __init__(self, dataset, model):
        super(FullCrownAdaptor, self).__init__(dataset, model)

    def compute(self, model, norm, x_U, x_L, eps, C):
        ub, _, lb, _ = model.full_backward_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)
        return ub, lb


class CrownIBPAdaptor(CrownAdaptorBase):

    def __init__(self, dataset, model):
        super(CrownIBPAdaptor, self).__init__(dataset, model)

    def compute(self, model, norm, x_U, x_L, eps, C):
        model.interval_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)
        ub, _, lb, _ = model.backward_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)
        return ub, lb


class IBPAdaptor(CrownAdaptorBase):

    def __init__(self, dataset, model):
        super(IBPAdaptor, self).__init__(dataset, model)

    def compute(self, model, norm, x_U, x_L, eps, C):
        ub, lb, _, _, _, _ = model.interval_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)
        return ub, lb
