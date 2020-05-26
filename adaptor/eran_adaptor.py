from constants import ELINA_PYTHON_INTERFACE_PATH, DEEPG_CODE_PATH
import sys
sys.path.insert(0, ELINA_PYTHON_INTERFACE_PATH)
sys.path.insert(0, DEEPG_CODE_PATH)

import numpy as np

import torch
import torch.nn as nn
import torch.onnx

import onnx
import onnxsim

from datasets import get_input_shape, get_num_classes
from adaptor.basic_adaptor import VerifierAdaptor

from eran.tf_verify.eran import ERAN
from eran.tf_verify.read_net_file import read_onnx_net
from eran.tf_verify.config import config
from eran.tf_verify.constraints import get_constraints_for_dominant_label
from eran.tf_verify.ai_milp import verify_network_with_milp


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d


#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

class ERANBase(VerifierAdaptor):

    def __init__(self, dataset, model):
        super(ERANBase, self).__init__(dataset, model)

        self.model.eval()

        # export the model to onnx file 'tmp/tmp.onnx'
        input_shape = get_input_shape(dataset)
        x = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], requires_grad=True).cuda()
        torch_out = self.model(x)

        torch.onnx.export(self.model,
                          x,
                          'tmp/tmp.onnx')
                          # export_params=True,
                          # opset_version=10,
                          # do_constant_folding=True,
                          # input_names=['input'],
                          # output_names=['output'],
                          # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

        model_opt, check_ok = onnxsim.simplify(
            'tmp/tmp.onnx', check_n=3, perform_optimization=True, skip_fuse_bn=True,
            input_shapes={None: (1, input_shape[0], input_shape[1], input_shape[2])}
        )
        assert check_ok, "Simplify ONNX model failed"

        onnx.save(model_opt, 'tmp/tmp_simp.onnx')
        new_model, is_conv = read_onnx_net('tmp/tmp_simp.onnx')
        self.new_model, self.is_conv = new_model, is_conv

        self.config = {}

        self.eran = ERAN(self.new_model, is_onnx=True)

    def load_config(self):
        """
            Make sure to run this before running each single verify()
        :return: None
        """
        for k in self.config:
            setattr(config, k, self.config[k])

    def verify(self, input, label, norm_type, radius):

        assert norm_type == 'inf'

        self.load_config()
        input = self.input_preprocess(input)
        m_radius = radius / self.coef

        image = input.permute(1, 2, 0).contiguous().numpy().reshape(-1)
        specLB = np.copy(image)
        specUB = np.copy(image)

        domain = config.domain
        pred,nn,nlb,nub = self.eran.analyze_box(
            specLB, specUB, init_domain(domain),
            config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
        if pred == label:
            specLB = np.clip(image - m_radius, self.in_min, self.in_max)
            specUB = np.clip(image + m_radius, self.in_min, self.in_max)

            perturbed_label, _, nlb, nub = self.eran.analyze_box(
                specLB, specUB, domain, config.timeout_lp,
                config.timeout_milp, config.use_default_heuristic)
            # print("nlb ", nlb[len(nlb) - 1], " nub ", nub[len(nub) - 1])
            if (perturbed_label == label):
                return True
            else:
                if config.complete:
                    # print("complete verification here")
                    constraints = get_constraints_for_dominant_label(label, get_num_classes(self.dataset))
                    verified_flag, adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                    if (verified_flag == True):
                        return True
                    else:
                        return False
                else:
                    return False

        else:
            raise Exception("Wrong prediction... This should not happen according to evaluate.py")
            # print('Prediction is wrong')
            return False


class AI2Adaptor(ERANBase):

    def __init__(self, dataset, model):
        super(AI2Adaptor, self).__init__(dataset, model)

        self.config = {
            'domain': 'deeppoly',
            'complete': True,
            'numproc': 20
        }


class DeepZonoAdaptor(ERANBase):

    def __init__(self, dataset, model):
        super(DeepZonoAdaptor, self).__init__(dataset, model)

        self.config = {
            'domain': 'deepzono',
            'complete': False,
            'numproc': 20
        }


class RefineZonoAdaptor(ERANBase):

    def __init__(self, dataset, model):
        super(RefineZonoAdaptor, self).__init__(dataset, model)

        self.config = {
            'domain': 'refinezono',
            'complete': False,
            'numproc': 20
        }


class DeepPolyAdaptor(ERANBase):

    def __init__(self, dataset, model):
        super(DeepPolyAdaptor, self).__init__(dataset, model)

        self.config = {
            'domain': 'deeppoly',
            'complete': False,
            'numproc': 20
        }


class RefinePolyAdaptor(ERANBase):

    def __init__(self, dataset, model):
        super(RefinePolyAdaptor, self).__init__(dataset, model)

        self.config = {
            'domain': 'refinepoly',
            'complete': False,
            'numproc': 20
        }


class KReluAdaptor(ERANBase):

    def __init__(self, dataset, model, domain='refinepoly', mode='dynamic'):
        super(KReluAdaptor, self).__init__(dataset, model)

        # === hyperparameter section ===
        assert domain in ['refinepoly', 'refinezono'], 'unsupported domain'
        assert mode == 2 or mode == 3 or mode == 'dynamic', 'unsupported mode'
        # === end of hyperparameter section ===

        # dynamic adjust the number of krelu
        self.config = {
            'domain': domain,
            'complete': False,
            'dyn_krelu': domain == 'dynamic',
            'use_2relu': domain == 2,
            'use_3relu': domain == 3,
            'numproc': 20
        }

