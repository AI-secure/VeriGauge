import torch
import torch.nn as nn

from adaptor.adaptor import Adaptor
from datasets import NormalizeLayer

from convex_adversarial.convex_adversarial import robust_loss



class ZicoDualAdaptor(Adaptor):
    """
        The adaptor for Zico's LP-dual approach
    """

    def __init__(self, dataset, model):
        super(ZicoDualAdaptor, self).__init__(dataset, model)
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

    def build_new_model(self):
        # the input is functioned as the canopy
        if isinstance(self.model[0], NormalizeLayer) and isinstance(self.model[1], nn.Sequential):
            # the model is concatenated with a normalized layer at the front
            # just like what we did in models/test_model.py
            self.new_model = self.model[1]
        else:
            self.new_model = self.model

    def input_preprocess(self, input):
        flayer = self.model[0]
        (_, height, width) = input.shape
        if isinstance(flayer, NormalizeLayer):
            input = (input - flayer.means.cpu().repeat(height, width, 1).permute(2, 0, 1)) / flayer.sds.cpu().repeat(height, width, 1).permute(2, 0, 1)
        return input

    def verify(self, input, label, norm_type, radius) -> bool:
        """
            The default implementation assumes the completeness of calc_radius() function
        :param input: input sample
        :param label: correct label
        :param norm_type: norm type
        :param radius: the floating point radius
        :return: bool
        """
        # super L2, Linf norm
        assert norm_type in ['inf', '2']
        norm = {
            'inf': 'l1',
            '2': 'l2'
        }[norm_type]

        m_radius = radius / self.coef

        X = self.input_preprocess(input).unsqueeze(0).cuda()
        y = torch.tensor([label]).cuda()

        self.new_model.eval()
        torch.set_grad_enabled(False)
        robust_ce, robust_err = robust_loss(self.new_model, m_radius, X, y, proj=False, norm_type=norm)
        torch.set_grad_enabled(True)
        # actually it could only be 0.0 or 1.0
        return robust_err < 0.5
