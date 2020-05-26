from time import time
import datetime
import sys
import torch.nn as nn
import tensorflow as tf
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import torch

from adaptor.adaptor import Adaptor
from cleverhans.attacks import ProjectedGradientDescent, CarliniWagnerL2
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

from datasets import NormalizeLayer, get_num_classes, get_input_shape

from basic.models import model_transform
from basic.intervalbound import IntervalFastLinBound, IntervalBound, FastIntervalBound
from basic.milp import MILPVerifier
from basic.fastmilp import FastMILPVerifier
from basic.percysdp import PercySDP
from basic.components import BaselinePointVerifierExt

# global graph
# graph = tf.get_default_graph()

class BasicAdaptor(Adaptor):
    """
        The adaptor for our basic framework
    """

    def __init__(self, dataset, model):
        super(BasicAdaptor, self).__init__(dataset, model)
        self.model = model
        # if the normalized layer exists,
        # we need to calculate the norm scaling coefficient
        # when the normalization layer is removed
        # real radius with normalization can thus be figured out
        self.coef = 1.0
        if isinstance(self.model[0], NormalizeLayer):
            self.coef = min(self.model[0].orig_sds)
        print(self.coef)

class CleanAdaptor(BasicAdaptor):
    """
        ** Not a real attack **
        Clean predictor
    """

    def verify(self, input, label, norm_type, radius):
        EPS = 1e-6
        # EPS = 0.02
        if radius > EPS:
            return False
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        return clean_pred == label


class PGDAdaptor(BasicAdaptor):
    """
        ** Not a real attack **
        For PGD attack, which only provides the lower bound for the robust radius
    """

    def __init__(self, dataset, model):
        super(PGDAdaptor, self).__init__(dataset, model)

        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
        self.config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.config)

        input_shape = get_input_shape(dataset)

        with self.sess.graph.as_default():
            with self.sess.as_default():
                self.tf_model = convert_pytorch_model_to_tf(self.model)
                self.ch_model = CallableModelWrapper(self.tf_model, output_layer='logits')

                self.x_op = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2],))
                self.attk = ProjectedGradientDescent(self.ch_model, sess=self.sess)

        self.adv_preds_ops = dict()

    def verify(self, input, label, norm_type, radius):

        # only support Linfty norm
        assert norm_type == 'inf'

        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False
        if radius == 0:
            return True

        with self.sess.graph.as_default():
            with self.sess.as_default():
                if radius not in self.adv_preds_ops:
                    params = {'eps': radius,
                              'clip_min': 0.0,
                              'clip_max': 1.0,
                              'eps_iter': radius / 50.0,
                              'nb_iter': 100,
                              'rand_init': False}
                    adv_x = self.attk.generate(self.x_op, **params)
                    self.adv_preds_ops[radius] = self.tf_model(adv_x)
                (adv_preds,) = self.sess.run((self.adv_preds_ops[radius],), feed_dict={self.x_op: xs})

                adv_pred = np.argmax(adv_preds[0])
        return adv_pred == label


class CWAdaptor(BasicAdaptor):
    """
        ** Not a real attack **
        For PGD attack, which only provides the lower bound for the robust radius
    """

    def __init__(self, dataset, model):
        super(CWAdaptor, self).__init__(dataset, model)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)

        self.tf_model = convert_pytorch_model_to_tf(self.model)
        self.ch_model = CallableModelWrapper(self.tf_model, output_layer='logits')

        self.dataset = dataset

    def calc_radius(self, input, label, norm_type, upper=0.5, eps=1e-2):

        # only support L2 norm
        assert norm_type == '2'

        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return 0.

        x_op = tf.placeholder(tf.float32, shape=(None, input.shape[0], input.shape[1], input.shape[2],))
        attk = CarliniWagnerL2(self.ch_model, sess=self.sess)
        params = {'y': tf.one_hot([label], get_num_classes(self.dataset)),
                  'clip_min': 0.0,
                  'clip_max': 1.0,
                  'max_iterations': 1000}
        adv_x = attk.generate(x_op, **params)
        adv_preds_op = self.tf_model(adv_x)

        (adv_preds, adv_xsamp) = self.sess.run((adv_preds_op, adv_x), feed_dict={x_op: xs})

        adv_pred = np.argmax(adv_preds[0])
        if adv_pred == label:
            # fail to find out adv example, return the radius to be the maximum one
            return la.norm(np.ones_like(adv_xsamp.reshape(-1)) * 0.5, 2)
        else:
            dist = la.norm(adv_xsamp.reshape(-1) - xs.numpy().reshape(-1), 2)
            return dist


class VerifierAdaptor(BasicAdaptor):
    """
        This adaptor covers the detail about the troublesome normalization
    """

    def __init__(self, dataset, model):
        super(VerifierAdaptor, self).__init__(dataset, model)
        self.flayer = None

        in_min, in_max = 0.0, 1.0
        if isinstance(model[0], NormalizeLayer):
            self.flayer, self.model = model
            in_min = (in_min - max(self.flayer.orig_means))
            in_max = (in_max - min(self.flayer.orig_means))
            if in_min <= 0.0:
                in_min = in_min / min(self.flayer.orig_sds)
            else:
                in_min = in_min / max(self.flayer.orig_sds)
            if in_max <= 0.0:
                in_max = in_max / max(self.flayer.orig_sds)
            else:
                in_max = in_max / min(self.flayer.orig_sds)
        self.in_min, self.in_max = in_min, in_max

    def input_preprocess(self, input):
        if self.flayer is not None:
            (_, height, width) = input.shape
            input = (input - self.flayer.means.cpu().repeat(height, width, 1).permute(2, 0, 1)) / self.flayer.sds.cpu().repeat(height, width, 1).permute(2, 0, 1)
        return input


class RealAdaptorBase(VerifierAdaptor):
    """
        Base class for real verification approaches.
        It deals with input transformation, model transformation, etc
        Note that in constructor, we assume the unnormalized data range is 0 ~ 1
    """

    def __init__(self, dataset, model):
        super(RealAdaptorBase, self).__init__(dataset, model)
        self.new_model = None
        self.bound = None

    def build_new_model(self, input):
        # the input is functioned as the canopy
        self.new_model = model_transform(self.model, list(input.shape))

    def prepare_solver(self, in_shape):
        raise NotImplementedError

    def verify(self, input, label, norm_type, radius):
        # only support Linfty norm
        assert norm_type == 'inf'
        if self.new_model is None:
            # init at the first time
            before = time()
            print(f"Init model for {self.__class__.__name__}...")
            in_shape = list(input.shape)
            self.build_new_model(input)
            self.prepare_solver(in_shape)
            after = time()
            print("Init done, time", str(datetime.timedelta(seconds=(after - before))),)

        # firstly check the clean prediction
        input = self.input_preprocess(input)
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False

        m_radius = radius / self.coef
        input = input.contiguous().view(-1)
        self.bound.calculate_bound(input, m_radius)

        for i in range(get_num_classes(self.dataset)):
            if i != label:
                ok = self.bound.verify(label, i)
                if not ok:
                    return False
        return True


class IBPAdaptor(VerifierAdaptor):
    """
        Interval Bound Propagation
    """
    def __init__(self, dataset, model):
        super(IBPAdaptor, self).__init__(dataset, model)
        in_shape = get_input_shape(dataset)
        self.bound = FastIntervalBound(self.model, in_shape, self.in_min, self.in_max)

    def verify(self, input, label, norm_type, radius):
        """
            Here we overwrite the base class verify() method
        """
        # only support Linfty norm
        assert norm_type == 'inf'

        # firstly check the clean prediction
        input = self.input_preprocess(input)
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False

        input = torch.Tensor(input)
        m_radius = radius / self.coef
        self.bound.calculate_bound(input, m_radius)

        for i in range(get_num_classes(self.dataset)):
            if i != label:
                ok = self.bound.verify(label, i)
                if not ok:
                    return False
        return True


class FastLinIBPAdaptor(RealAdaptorBase):
    """
        fastlin with Interval Bound Improvement
    """

    def prepare_solver(self, in_shape):
        self.bound = IntervalFastLinBound(self.new_model, in_shape, self.in_min, self.in_max)


class MILPAdaptor(RealAdaptorBase):
    """
        MILP from Tjeng et al
    """

    def __init__(self, dataset, model, timeout=30):
        super(MILPAdaptor, self).__init__(dataset, model)
        self.timeout = timeout

    def prepare_solver(self, in_shape):
        self.prebound = IntervalFastLinBound(self.new_model, in_shape, self.in_min, self.in_max)
        self.bound = MILPVerifier(self.new_model, in_shape, self.in_min, self.in_max)

    def verify(self, input, label, norm_type, radius):
        """
            Here we overwrite the base class verify() method
        """
        # only support Linfty norm
        assert norm_type == 'inf'
        if self.new_model is None:
            # init at the first time
            before = time()
            print(f"Init model for {self.__class__.__name__}...")
            in_shape = list(input.shape)
            self.build_new_model(input)
            self.prepare_solver(in_shape)
            after = time()
            print("Init done, time", str(datetime.timedelta(seconds=(after - before))), )

        # firstly check the clean prediction
        input = self.input_preprocess(input)
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False

        m_radius = radius / self.coef

        input = input.contiguous().view(-1)
        self.prebound.calculate_bound(input, m_radius)
        self.bound.construct(self.prebound.l, self.prebound.u, input, m_radius)

        for i in range(get_num_classes(self.dataset)):
            if i != label:
                self.bound.prepare_verify(label, i)
                try:
                    # self.bound.prob.solve(verbose=True)
                    # model.setParam(GRB.Param.TimeLimit, timeout)
                    self.bound.prob.solve(solver=cp.GUROBI, verbose=False, BestObjStop=-1e-6, TimeLimit=70, Threads=20)
                except:
                    return False
                print(self.bound.prob.status, self.bound.prob.value, file=sys.stderr)
                if self.bound.prob.status not in ['optimal'] or self.bound.prob.value < 0.:
                    return False
        return True



class FastMILPAdaptor(RealAdaptorBase):
    """
        MILP from Tjeng et al
    """

    def __init__(self, dataset, model, timeout=30):
        super(FastMILPAdaptor, self).__init__(dataset, model)
        # cp.settings.SOLVE_TIME = timeout
        in_shape = get_input_shape(dataset)
        self.prepare_solver(in_shape)

    def prepare_solver(self, in_shape):
        self.prebound = FastIntervalBound(self.model, in_shape, self.in_min, self.in_max)
        self.bound = FastMILPVerifier(self.model, in_shape, self.in_min, self.in_max)

    def verify(self, input, label, norm_type, radius):
        """
            Here we overwrite the base class verify() method
        """
        # only support Linfty norm
        assert norm_type == 'inf'
        # if self.new_model is None:
        #     # init at the first time
        #     before = time()
        #     print(f"Init model for {self.__class__.__name__}...")
        #     in_shape = list(input.shape)
        #     # self.build_new_model(input)
        #     self.prepare_solver(in_shape)
        #     after = time()
        #     print("Init done, time", str(datetime.timedelta(seconds=(after - before))), )

        # firstly check the clean prediction
        input = self.input_preprocess(input)
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False

        m_radius = radius / self.coef

        # input_cons = input.contiguous().view(-1)
        self.prebound.calculate_bound(input, m_radius)
        self.bound.construct(self.prebound.l, self.prebound.u, input, m_radius)

        for i in range(get_num_classes(self.dataset)):
            if i != label:
                if not self.bound.verify(label, i):
                    return False
                # try:
                # self.bound.prob.solve(verbose=True)
                # model.setParam(GRB.Param.TimeLimit, timeout)
                # self.bound.prob.solve(solver=cp.GUROBI, verbose=False, BestObjStop=-1e-6, TimeLimit=70, Threads=20)
                # except:
                #     return False
                # print(self.bound.prob.status, self.bound.prob.value, file=sys.stderr)
                # if self.bound.prob.status not in ['optimal'] or self.bound.prob.value < 0.:
                #     return False
                # return False
        return True



class PercySDPAdaptor(RealAdaptorBase):
    """
        SDP from Percy et al
    """

    def __init__(self, dataset, model, timeout=30):
        super(PercySDPAdaptor, self).__init__(dataset, model)
        cp.settings.SOLVE_TIME = timeout

    def prepare_solver(self, in_shape):
        self.prebound = IntervalFastLinBound(self.new_model, in_shape, self.in_min, self.in_max)
        self.bound = PercySDP(self.new_model, in_shape)

    def verify(self, input, label, norm_type, radius):
        """
            Here we overwrite the base class verify() method
        """
        # only support Linfty norm
        assert norm_type == 'inf'
        if self.new_model is None:
            # init at the first time
            before = time()
            print(f"Init model for {self.__class__.__name__}...")
            in_shape = list(input.shape)
            self.build_new_model(input)
            self.prepare_solver(in_shape)
            after = time()
            print("Init done, time", str(datetime.timedelta(seconds=(after - before))), )

        # firstly check the clean prediction
        input = self.input_preprocess(input)
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False

        m_radius = radius / self.coef

        input = input.contiguous().view(-1)
        self.prebound.calculate_bound(input, m_radius)
        bl = [np.maximum(self.prebound.l[i], 0) if i > 0 else self.prebound.l[i] for i in range(len(self.prebound.l))]
        bu = [np.maximum(self.prebound.u[i], 0) if i > 0 else self.prebound.u[i] for i in range(len(self.prebound.u))]

        for i in range(get_num_classes(self.dataset)):
            if i != label:
                self.bound.run(bl, bu, label, i)
                if self.bound.prob.status not in ['optimal'] or self.bound.prob.value > 0.:
                    return False
        return True


class FazlybSDPAdaptor(RealAdaptorBase):
    """
        SDP from Fazlyb et al
    """

    def __init__(self, dataset, model, timeout=30):
        super(FazlybSDPAdaptor, self).__init__(dataset, model)
        cp.settings.SOLVE_TIME = timeout

    def prepare_solver(self, in_shape):
        self.prebound = IntervalFastLinBound(self.new_model, in_shape, self.in_min, self.in_max)

    def verify(self, input, label, norm_type, radius):
        """
            Here we overwrite the base class verify() method
        """
        in_shape = list(input.shape)
        # only support Linfty norm
        assert norm_type == 'inf'
        if self.new_model is None:
            # init at the first time
            before = time()
            print(f"Init model for {self.__class__.__name__}...")
            self.build_new_model(input)
            self.prepare_solver(in_shape)
            after = time()
            print("Init done, time", str(datetime.timedelta(seconds=(after - before))), )

        # firstly check the clean prediction
        input = self.input_preprocess(input)
        xs = input.unsqueeze(0)
        clean_preds = self.model(xs.cuda()).detach().cpu().numpy()
        clean_pred = np.argmax(clean_preds[0])
        if clean_pred != label:
            return False

        m_radius = radius / self.coef

        input = input.contiguous().view(-1)
        self.prebound.calculate_bound(input, m_radius)
        bl = [np.maximum(self.prebound.l[i], 0) if i > 0 else self.prebound.l[i] for i in range(len(self.prebound.l))]
        bu = [np.maximum(self.prebound.u[i], 0) if i > 0 else self.prebound.u[i] for i in range(len(self.prebound.u))]

        pv = BaselinePointVerifierExt(self.new_model, in_shape, self.in_min, self.in_max)


        for i in range(get_num_classes(self.dataset)):
            if i != label:

                pv.create_cmat(input, label, i, m_radius, bl, bu)
                pv.run()

                if pv.prob.status not in ['unbounded', 'unbounded_inaccurate'] and pv.prob.value > 0.:
                    return False
        return True
