import torch
from torch import nn

import sys
import cvxpy as cp
import numpy as np

from models.zoo import Flatten
from basic.models import FlattenConv2D
from .core import *

class BaselinePointVerifier():

    def __init__(self, model, in_shape, in_min, in_max, create_opt_vars=True, large_gamma=False, large_t=False):
        super(BaselinePointVerifier, self).__init__()

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
        self.tot_n = sum(self.shapes[:-1]) + 1
        self.Ws = Ws
        self.bs = bs

        self.gamma = None
        self.lbdas = None
        self.nus = None
        self.etas = None
        self.s = None

        self.constraints = list()

        if create_opt_vars:
            self.create_opt_vars(large_gamma, large_t)

        self.cmat = None

        self.x0 = None
        self.y0 = None
        self.yp = None
        self.eps = None

        self.prob = None

    def create_opt_vars(self, large_gamma=False, large_t=False):
        # init gamma from abs(N(0,1))
        if large_gamma:
            gamma = cp.Variable((self.in_numel, self.in_numel), name='gamma')
        else:
            gamma = cp.Variable((self.in_numel,), name='gamma')
        self.constraints.append((gamma >= 0.))

        s = cp.Variable(name='s')

        if large_t:
            lbda_n = sum(self.shapes[1:-1])
            lbda_vec = cp.Variable((lbda_n,), name='lbda_vec')
            lbda_mat = cp.Variable((lbda_n, lbda_n), name='lbda_mat', symmetric=True)
            self.constraints.append((lbda_mat >= 0.))
            lbda_mat_masked = cp.multiply((- np.ones((lbda_n, lbda_n)) + np.eye(lbda_n)), lbda_mat)
            lbda_mat_diag = cp.hstack([- cp.sum(lbda_mat_masked[i, i+1:]) for i in range(lbda_n)])
            lbdas = cp.diag(lbda_vec) + lbda_mat_masked + cp.diag(lbda_mat_diag)
        else:
            lbdas = list()

        nus = list()
        etas = list()

        for i in range(1, len(self.shapes) - 1):
            if not large_t:
                now_lbda = cp.Variable((self.shapes[i],), name=f'lbda_{i}')
            now_nu = cp.Variable((self.shapes[i],), name=f'nu_{i}')
            now_eta = cp.Variable((self.shapes[i],), name=f'eta_{i}')

            if not large_t:
                self.constraints.append((now_lbda >= 0.))
            self.constraints.append((now_nu >= 0.))
            self.constraints.append((now_eta >= 0.))

            if not large_t:
                lbdas.append(now_lbda)
            nus.append(now_nu)
            etas.append(now_eta)

        self.gamma = gamma
        self.lbdas = lbdas
        self.nus = nus
        self.etas = etas
        self.s = s

    def create_cmat(self, x0, y0, yp, eps, indirect=False):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(y0, torch.Tensor):
            y0 = y0.cpu().numpy()
        if isinstance(yp, torch.Tensor):
            yp = yp.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()
        if self.cmat is not None:
            del self.cmat
        self.x0 = x0
        self.y0 = y0
        self.yp = yp
        self.eps = eps
        if indirect or not isinstance(self.lbdas, list):
            self.cmat = calc_full_cp_matrix_indirect(
                [self.gamma, self.lbdas, self.nus, self.etas, self.s],
                [self.Ws, self.bs],
                self.x0,
                self.y0,
                self.yp,
                self.eps,
                self.shapes,
                self.in_min,
                self.in_max
            )
        else:
            self.cmat = calc_full_cp_matrix(
                [self.gamma, self.lbdas, self.nus, self.etas, self.s],
                [self.Ws, self.bs],
                self.x0,
                self.y0,
                self.yp,
                self.eps,
                self.shapes,
                self.in_min,
                self.in_max
            )

    def run(self):
        self.constraints.append((self.cmat << 0))
        self.prob = cp.Problem(cp.Minimize(self.s), self.constraints)
        self.prob.solve(solver=cp.SCS, verbose=True)
        print('status:', self.prob.status)
        print('optimial value:', self.prob.value)


class BaselinePointVerifierExt():
    """
        Extended by upper and lower bounds
    """
    def __init__(self, model, in_shape, in_min, in_max, create_opt_vars=True, timeout=50, threads=30):
        super(BaselinePointVerifierExt, self).__init__()

        self.timeout, self.threads = timeout, threads

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
        self.tot_n = sum(self.shapes[:-1]) + 1
        self.Ws = Ws
        self.bs = bs

        self.gammas = None
        self.lbdas = None
        self.nus = None
        self.etas = None
        # self.xi = None
        self.s = None

        self.constraints = list()

        if create_opt_vars:
            self.create_opt_vars()

        self.cmat = None

        self.x0 = None
        self.y0 = None
        self.yp = None
        self.eps = None

        self.prob = None

    def create_opt_vars(self):

        s = cp.Variable(name='s')

        # xi = cp.Variable(name='xi')
        # self.constraints.append((xi >= 0.))

        gammas = list()
        lbdas = list()
        nus = list()
        etas = list()

        for i in range(0, len(self.shapes) - 1):

            now_gamma = cp.Variable((self.shapes[i],), name=f'gamma_{i}')
            self.constraints.append((now_gamma >= 0.))
            gammas.append(now_gamma)

            if i > 0:
                now_lbda = cp.Variable((self.shapes[i],), name=f'lbda_{i}')
                now_nu = cp.Variable((self.shapes[i],), name=f'nu_{i}')
                now_eta = cp.Variable((self.shapes[i],), name=f'eta_{i}')

                # self.constraints.append((now_lbda >= 0.))
                self.constraints.append((now_nu >= 0.))
                self.constraints.append((now_eta >= 0.))

                lbdas.append(now_lbda)
                nus.append(now_nu)
                etas.append(now_eta)

        self.gammas = gammas
        self.lbdas = lbdas
        self.nus = nus
        self.etas = etas
        # self.xi = xi
        self.s = s

    def create_cmat(self, x0, y0, yp, eps, l, u):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(y0, torch.Tensor):
            y0 = y0.cpu().numpy()
        if isinstance(yp, torch.Tensor):
            yp = yp.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()
        if self.cmat is not None:
            del self.cmat
        self.x0 = x0
        self.y0 = y0
        self.yp = yp
        self.eps = eps
        self.l = l
        self.u = u
        self.cmat = calc_full_ext_cp_matrix(
            [self.gammas, self.lbdas, self.nus, self.etas, None, self.s],
            [self.Ws, self.bs],
            self.x0,
            self.y0,
            self.yp,
            self.eps,
            self.l,
            self.u,
            self.shapes,
            self.in_min,
            self.in_max
        )

    def run(self):
        self.constraints.append((self.cmat << 0))
        self.prob = cp.Problem(cp.Minimize(self.s), self.constraints)
        self.prob.solve(solver=cp.SCS, verbose=True, warm_start=True, eps=1e-3)
        # self.prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={
        #     # 'optimizerMaxTime': self.timeout,
        #     'MSK_DPAR_OPTIMIZER_MAX_TIME': self.timeout,
        #     # 'numThreads': self.threads,
        #     'MSK_IPAR_NUM_THREADS': self.threads,
        #     # 'lowerObjCut': 0.,
        #     'MSK_DPAR_LOWER_OBJ_CUT': 0.,
        # })
        print('status:', self.prob.status)
        print('optimial value:', self.prob.value)


class PointVerifier(nn.Module):

    def __init__(self, model, in_shape, create_opt_vars=True):
        super(PointVerifier, self).__init__()

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
                        Ws.append(l.weight)
                        bs.append(l.bias)
                    elif isinstance(l, nn.Linear):
                        assert shapes[-1] == l.in_features
                        now_shape = l.out_features
                        Ws.append(l.weight)
                        bs.append(l.bias)
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

        self.gammas = None
        self.lbdas = None
        self.nus = None
        self.etas = None
        self.s = None

        # self.pos_gamma = None
        # self.pos_nus = None
        # self.pos_etas = None

        if create_opt_vars:
            self.create_opt_vars()

        self.smat = None
        self.mat = None
        self.symmat = None

        self.x0 = None
        self.y0 = None
        self.yp = None
        self.l = None
        self.u = None

        # positive
        self.pstv = True

    def create_opt_vars(self):
        # init s = 0, will adjust before every run
        s = torch.nn.Parameter(torch.tensor(0., dtype=torch.double))

        lbdas = list()
        nus = list()
        etas = list()
        for i in range(1, len(self.shapes) - 1):
            # lambda init from N(0,1)
            now_lbda = torch.nn.Parameter(torch.randn(self.shapes[i], dtype=torch.double) / self.shapes[i])
            # nu init from abs(N(0,1))
            now_nu = torch.nn.Parameter(torch.abs(torch.randn(self.shapes[i], dtype=torch.double) / self.shapes[i]))
            # eta init from abs(N(0,1))
            now_eta = torch.nn.Parameter(torch.abs(torch.randn(self.shapes[i], dtype=torch.double) / self.shapes[i]))
            self.register_parameter(f'lbda_{i}', now_lbda)
            self.register_parameter(f'nu_{i}', now_nu)
            self.register_parameter(f'eta_{i}', now_eta)
            lbdas.append(now_lbda)
            nus.append(now_nu)
            etas.append(now_eta)

        gammas = list()
        for i in range(0, len(self.shapes) - 1):
            # gamma_i init from abs(N(0,1))
            now_gamma = torch.nn.Parameter(torch.abs(torch.randn(self.shapes[i], dtype=torch.double) / self.shapes[i]))
            self.register_parameter(f'gamma_{i}', now_gamma)
            gammas.append(now_gamma)

        self.gammas = gammas
        self.lbdas = lbdas
        self.nus = nus
        self.etas = etas
        self.s = s

        self.matinv = None

        self.t = 0.

    def bump_opt_vers(self, gammas, lbdas, nus, etas, s):

        for i in range(0, len(self.shapes) - 1):
            assert gammas[i].size() == torch.Size([self.shapes[i]])

        for i in range(1, len(self.shapes) - 1):
            assert lbdas[i-1].size() == torch.Size([self.shapes[i]])
            assert nus[i-1].size() == torch.Size([self.shapes[i]])
            assert etas[i-1].size() == torch.Size([self.shapes[i]])

        self.gammas, self.lbdas, self.nus, self.etas, self.s = gammas, lbdas, nus, etas, s

    def update_s(self, s):

        if isinstance(s, float):
            if self.s.is_cuda:
                self.s.data = torch.tensor(s, dtype=torch.double).cuda()
            else:
                self.s.data = torch.tensor(s, dtype=torch.double)
        elif isinstance(s, torch.Tensor):
            if self.s.is_cuda:
                self.s.data = s.clone().cuda()
            else:
                self.s.data = s.clone()
        else:
            raise Exception(f'Unsupported s type: {type(s)}')

        # only update those that exist
        if self.smat is not None:
            self._create_smat()
        if self.mat is not None:
            self._create_mat()
        if self.symmat is not None:
            self._create_symmat()

    def _relu(self, x):
        if isinstance(x, torch.Tensor):
            return torch.relu(x)
        elif isinstance(x, list):
            return [torch.relu(xx) for xx in x]
        else:
            raise Exception(f'Wrong data type given: {type(x)}')

    # symbolic matrix
    def _create_smat(self):
        if self.smat is not None:
            del self.smat
        self.smat = CompressMatrix(
            [self._relu(self.gammas), self.lbdas, self._relu(self.nus), self._relu(self.etas), self.s],
            [self.Ws, self.bs],
            self.x0,
            self.y0,
            self.yp,
            self.l,
            self.u,
            self.shapes)

    def _create_mat(self):
        if self.mat is not None:
            del self.mat
        self.mat = calc_full_matrix(
            [self._relu(self.gammas), self.lbdas, self._relu(self.nus), self._relu(self.etas), self.s],
            [self.Ws, self.bs],
            self.x0,
            self.y0,
            self.yp,
            self.l,
            self.u,
            self.shapes)

    def _create_symmat(self):
        if self.symmat is not None:
            del self.symmat
        self.symmat = calc_full_matrix(
            [self._relu(self.gammas), self.lbdas, self._relu(self.nus), self._relu(self.etas), self.s],
            [self.Ws, self.bs],
            self.x0,
            self.y0,
            self.yp,
            self.l,
            self.u,
            self.shapes,
            numerical=False)

    def create_mats(self, x0, y0, yp, l, u, incl='smat,mat,symmat'):

        self.x0, self.y0, self.yp, self.l, self.u = x0, y0, yp, l, u

        incl = incl.split(',')

        if 'smat' in incl:
            # compact representation matrix
            self._create_smat()

        if 'mat' in incl:
            # numerical torch.tensor matrix
            self._create_mat()

        if 'symmat' in incl:
            # symbolic torch.tensor matrix
            self._create_symmat()

    def max_eigen(self, method='power'):
        assert method in ['torch', 'power']
        if method == 'torch':
            # kinda time consuming, but accurate
            assert self.mat is not None
            eig, _ = torch.eig(self.mat)
            eig = eig[:, 0]
            max_eig = eig.max().item()
        else:
            # use power method
            assert self.smat is not None
            max_eig = self.smat.max_eigenvalue()
        return max_eig

    def inverse(self, method='torch'):
        assert method in ['cholesky', 'torch', 'compress_inverse']
        if method == 'cholesky':
            u = torch.cholesky(-self.mat)
            matinv = - torch.cholesky_inverse(u)
        elif method == 'torch':
            matinv = torch.inverse(self.mat)
        else:
            # compress inverse
            matinv = self.smat.inverse()
        self.matinv = matinv
        return matinv

    def forward(self, method='direct', use_det=False):
        assert method in ['torch', 'direct']
        if method == 'torch':
            loss = self.s * self.t + torch.sum(-self.matinv.cuda().t() * self.symmat)
            loss.backward(retain_graph=True)
            return loss
        else:
            with torch.no_grad():
                N = -self.matinv.cuda().t()

                shapes = self.shapes

                ptr = 0
                for i in range(len(self.gammas)):
                    preptr = ptr
                    ptr += shapes[i]
                    now_gamma_grad = - 2.0 * N.diag()[preptr: ptr] \
                                     + 2.0 * (torch.tensor(self.l[i] + self.u[i]).cuda() if N.is_cuda else torch.tensor(self.l[i] + self.u[i])) * N[preptr: ptr, -1] \
                                     - 2.0 * (torch.tensor(self.l[i] * self.u[i]).cuda() if N.is_cuda else torch.tensor(self.l[i] * self.u[i])) * N[-1, -1]
                    self.gammas[i].grad = now_gamma_grad.detach() * (self.gammas[i] > 0).double()

                ptr = 0
                for i in range(len(self.lbdas)):
                    preptr = ptr
                    ptr += shapes[i]
                    now_lbda_grad = + 2.0 * (self.Ws[i] * N[ptr: ptr + shapes[i+1], preptr: ptr]).sum(dim=-1) \
                                    + 2.0 * self.bs[i] * N[ptr: ptr + shapes[i+1], -1] \
                                    - 2.0 * N.diag()[ptr: ptr + shapes[i+1]]
                    self.lbdas[i].grad = now_lbda_grad.detach()
                    now_nu_grad = - 2.0 * torch.matmul(self.Ws[i], N[preptr: ptr, -1]) \
                                  + 2.0 * N[ptr: ptr + shapes[i+1], -1]\
                                  - 2.0 * self.bs[i] * N[-1, -1]
                    self.nus[i].grad = now_nu_grad.detach() * (self.nus[i] > 0.).double()
                    now_eta_grad = 2.0 * N[ptr: ptr + shapes[i+1], -1]
                    self.etas[i].grad = now_eta_grad.detach() * (self.etas[i] > 0.).double()

                self.s.grad = - torch.trace(N).detach() + self.t

                # time consuming!
                if use_det:

                    eigens, _ = torch.symeig(self.mat)
                    maxeigen = torch.max(eigens).item()
                    # print(maxeigen)

                    sgn = torch.tensor(1.0) if maxeigen <= -1e-6 else torch.tensor(-1.0)


                    # t1 = time.time()
                    # det = torch.det(self.mat)
                    # print(det)
                    # det = ((det < -1e-20).double() - 0.5) * 2.0
                    # t2 = time.time()
                    # # print(t2 - t1)
                else:
                    sgn = torch.tensor(1.0)

                self.pstv = (sgn >= 0.0)
                if use_det and maxeigen > 1e-6:
                    print('!!!!!!', file=sys.stderr)
                    # raise Exception('MMP!')

                # regularize grad by t to prevent booming
                self.s.grad /= self.t * sgn
                for i in range(len(self.gammas)):
                    self.gammas[i].grad /= self.t * sgn
                for param_set in [self.lbdas, self.nus, self.etas]:
                    for item in param_set:
                        item.grad /= self.t * sgn
                self.s.grad.clamp_(max=1.0)

                loss = self.s * self.t + torch.sum(-self.matinv.cuda().t() * self.symmat)
                return loss
