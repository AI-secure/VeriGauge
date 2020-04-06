import torch
import torch.nn as nn

from adaptor.basic_adaptor import VerifierAdaptor
from datasets import NormalizeLayer

from convex_adversarial.convex_adversarial import robust_loss



class ZicoDualAdaptor(VerifierAdaptor):
    """
        The adaptor for Zico's LP-dual approach
    """

    def __init__(self, dataset, model):
        super(ZicoDualAdaptor, self).__init__(dataset, model)

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

        self.model.eval()
        torch.set_grad_enabled(False)
        robust_ce, robust_err = robust_loss(self.model, m_radius, X, y, proj=False, norm_type=norm)
        torch.set_grad_enabled(True)
        # actually it could only be 0.0 or 1.0
        return robust_err < 0.5
