import time
import torch
import numpy as np
import cvxpy as cp

def calc_full_cp_matrix_indirect(opt_vars, weights, x0, y0, yp, eps, shapes, in_min, in_max):
    gamma, lbdas, nus, etas, s = opt_vars
    Ws, bs = weights

    x_min = np.clip(x0 - eps, a_min=in_min, a_max=in_max)
    x_max = np.clip(x0 + eps, a_min=in_min, a_max=in_max)

    A = [[Ws[i] if i == j else np.zeros((shapes[i+1], shapes[j])) for j in range(len(Ws))] for i in range(len(Ws) - 1)]
    B = [[np.eye(shapes[j]) if i + 1 == j else np.zeros((shapes[i+1], shapes[j])) for j in range(len(shapes) - 1)] for i in range(len(shapes) - 2)]
    C = [np.zeros((shapes[-1], shapes[i])) if i < len(Ws) - 1 else Ws[i] for i in range(len(Ws))]
    b = [bs[:-1]]

    A = np.block(A)
    B = np.block(B)
    C = np.block(C)
    b = np.block(b).T

    M_in_comp = [
        [np.eye(shapes[i]) if i == 0 else np.zeros((shapes[0], shapes[i] if i < len(shapes) - 1 else 1)) for i in range(len(shapes))],
        [np.zeros((1, shapes[i])) if i < len(shapes) - 1 else np.ones((1, 1)) for i in range(len(shapes))]
    ]
    M_mid_comp = [
        [A, b],
        [B, np.zeros((B.shape[0], 1))],
        [np.zeros((1, B.shape[1])), np.ones((1, 1))]
    ]
    M_out_comp = [
        [C, np.expand_dims(bs[-1], axis=1)],
        [np.zeros((1, C.shape[1])), np.ones((1, 1))]
    ]

    M_in_comp = np.bmat(M_in_comp)
    M_mid_comp = np.bmat(M_mid_comp)
    M_out_comp = np.bmat(M_out_comp)

    if gamma.ndim == 1:
        strip = cp.multiply(gamma, x_min + x_max)
        P = cp.bmat([
            [-cp.diag(gamma) * 2.0, cp.reshape(strip, (shapes[0], 1))],
            [cp.reshape(strip, (1, shapes[0])), cp.reshape(- (x_min * x_max) @ gamma * 2.0, (1, 1))]
        ])
    elif gamma.ndim == 2:
        strip = cp.reshape(gamma @ x_min + gamma.T @ x_max, (shapes[0], 1))
        P = cp.bmat([
            [- gamma - gamma.T, strip],
            [strip.T, cp.reshape(- (x_min.reshape((1, shapes[0])) @ gamma.T @ x_max.reshape((shapes[0], 1))) - (x_max.reshape((1, shapes[0])) @ gamma @ x_min.reshape((shapes[0], 1))), (1, 1))]
        ])

    if isinstance(lbdas, list):
        T = cp.diag(cp.vec(cp.hstack(lbdas)))
    else:
        T = lbdas
    nu_vec = cp.vec(cp.hstack(nus))
    eta_vec = cp.vec(cp.hstack(etas))
    block_n = nu_vec.shape[0]
    Q = cp.bmat([
        [np.zeros((block_n, block_n)), T, -cp.reshape(nu_vec, (block_n, 1))],
        [T, -2.0 * T, cp.reshape((nu_vec + eta_vec), (block_n, 1))],
        [-cp.reshape(nu_vec, (1, block_n)), cp.reshape((nu_vec + eta_vec), (1, block_n)), np.zeros((1, 1))]
    ])

    ai = np.zeros((shapes[-1], 1))
    ai[y0, 0] = -1
    ai[yp, 0] = 1
    S = cp.bmat([
        [np.zeros((shapes[-1], shapes[-1])), ai],
        [ai.T, np.zeros((1, 1))]
    ])

    mat = M_in_comp.T @ P @ M_in_comp + M_mid_comp.T @ Q @ M_mid_comp + M_out_comp.T @ S @ M_out_comp - s * np.eye(sum(shapes) - shapes[-1] + 1)
    return mat


def calc_full_ext_cp_matrix(opt_vars, weights, x0, y0, yp, eps, l, u, shapes, in_min, in_max):
    gammas, lbdas, nus, etas, _, s = opt_vars
    Ws, bs = weights

    first_ele = - 2.0 * gammas[0] - s

    last_ele = - s \
               - cp.sum(cp.hstack([x@y for (x, y) in zip(nus, bs)])) * 2.0 \
               - cp.sum(cp.hstack([gammas[i] @ (l[i] * u[i] * 2.0) for i in range(len(gammas))])) \
               + (-bs[-1][y0] + bs[-1][yp]) * 2.0 # + xi

    diag = [first_ele] + [- lbdas[i] * 2.0 - 2.0 * gammas[i + 1] - s for i in range(len(lbdas))] + [last_ele]
    diag = cp.hstack(diag)

    vecs = list()
    for i in range(len(Ws)):
        local_vec = cp.multiply(gammas[i], l[i] + u[i])
        if i < len(Ws) - 1:
            local_vec = local_vec - Ws[i].T @ nus[i]
        else:
            local_vec = local_vec - Ws[i][y0] + Ws[i][yp]
        if i > 0:
            local_vec = local_vec + nus[i-1] + etas[i-1] + cp.multiply(lbdas[i-1], bs[i-1])
        vecs.append(local_vec)

    submat = list()
    for i in range(len(Ws) - 1):
        submat.append(cp.diag(lbdas[i]) * Ws[i])

    now_blockmat = list()
    ni = 0
    for i in range(len(shapes)):
        nj = 0
        now_matlist = list()
        wr = shapes[i] if i < len(shapes) - 1 else 1
        for j in range(len(shapes)):
            wc = shapes[j] if j < len(shapes) - 1 else 1
            if i < len(shapes) - 1 and j < len(shapes) - 1:
                if j < i - 1:
                    now_matlist.append(np.zeros((wr, wc)))
                elif j == i - 1:
                    now_matlist.append(submat[j])
                elif j == i:
                    now_matlist.append(cp.diag(diag[ni: ni + wr]))
                elif j == i + 1:
                    now_matlist.append(submat[i].T)
                elif j > i + 1:
                    now_matlist.append(np.zeros((wr, wc)))
            elif i == len(shapes) - 1 and j < len(shapes) - 1:
                now_matlist.append(cp.reshape(vecs[j], (1, wc)))
            elif i < len(shapes) - 1 and j == len(shapes) - 1:
                now_matlist.append(cp.reshape(vecs[i], (wr, 1)))
            else:
                now_matlist.append(cp.reshape(last_ele, (1, 1)))
            nj += wc
        ni += wr

        now_blockmat.append(now_matlist)

    mat = cp.bmat(now_blockmat)
    return mat


def calc_full_cp_matrix(opt_vars, weights, x0, y0, yp, eps, shapes, in_min, in_max):
    gamma, lbdas, nus, etas, s = opt_vars
    Ws, bs = weights

    x_min = np.clip(x0 - eps, a_min=in_min, a_max=in_max)
    x_max = np.clip(x0 + eps, a_min=in_min, a_max=in_max)

    if gamma.ndim == 1:
        x_minmax = x_min * x_max

        first_ele = - gamma * 2.0 - s

        last_ele = - s \
                   - cp.sum(cp.hstack([x@y for (x, y) in zip(nus, bs)])) * 2.0 \
                   - gamma @ x_minmax * 2.0 \
                   + (-bs[-1][y0] + bs[-1][yp]) * 2.0

        diag = [first_ele] + [- l * 2.0 - s for l in lbdas] + [last_ele]
        diag = cp.hstack(diag)

    elif gamma.ndim == 2:

        first_ele = - gamma - gamma.T - s

        x_min_vec = x_min.reshape(-1, 1)
        x_max_vec = x_max.reshape(-1, 1)

        last_ele = - s \
                   - cp.sum(cp.hstack([x@y for (x, y) in zip(nus, bs)])) * 2.0 \
                   - (x_min_vec.T @ gamma.T @ x_max_vec) - (x_max_vec.T @ gamma @ x_min_vec) \
                   + (-bs[-1][y0] + bs[-1][yp]) * 2.0

        diag = [first_ele] + [cp.diag(- l * 2.0 - s) for l in lbdas] + [last_ele]

    vecs = list()
    for i in range(len(Ws)):
        local_vec = np.zeros((Ws[i].shape[1],))
        if i == 0:
            if gamma.ndim == 1:
                local_vec = local_vec + cp.multiply(gamma, x_min + x_max)
            elif gamma.ndim == 2:
                local_vec = local_vec + gamma @ x_min + gamma.T @ x_max
        if i < len(Ws) - 1:
            local_vec = local_vec - Ws[i].T @ nus[i]
        else:
            local_vec = local_vec - Ws[i][y0] + Ws[i][yp]
        if i > 0:
            local_vec = local_vec + nus[i-1] + etas[i-1] + cp.multiply(lbdas[i-1], bs[i-1])
        vecs.append(local_vec)

    submat = list()
    for i in range(len(Ws) - 1):
        submat.append(cp.diag(lbdas[i]) * Ws[i])

    now_blockmat = list()
    ni = 0
    for i in range(len(shapes)):
        nj = 0
        now_matlist = list()
        wr = shapes[i] if i < len(shapes) - 1 else 1
        for j in range(len(shapes)):
            wc = shapes[j] if j < len(shapes) - 1 else 1
            if i < len(shapes) - 1 and j < len(shapes) - 1:
                if j < i - 1:
                    now_matlist.append(np.zeros((wr, wc)))
                elif j == i - 1:
                    now_matlist.append(submat[j])
                elif j == i:
                    if gamma.ndim == 1:
                        now_matlist.append(cp.diag(diag[ni: ni + wr]))
                    else:
                        now_matlist.append(diag[i])
                elif j == i + 1:
                    now_matlist.append(submat[i].T)
                elif j > i + 1:
                    now_matlist.append(np.zeros((wr, wc)))
            elif i == len(shapes) - 1 and j < len(shapes) - 1:
                now_matlist.append(cp.reshape(vecs[j], (1, wc)))
            elif i < len(shapes) - 1 and j == len(shapes) - 1:
                now_matlist.append(cp.reshape(vecs[i], (wr, 1)))
            else:
                if gamma.ndim == 1:
                    now_matlist.append(cp.reshape(last_ele, (1, 1)))
                elif gamma.ndim == 2:
                    now_matlist.append(diag[i])
            nj += wc
        ni += wr

        now_blockmat.append(now_matlist)

    mat = cp.bmat(now_blockmat)
    return mat


def calc_full_matrix(opt_vars, weights, x0, y0, yp, l, u, shapes, numerical=True):
    """
        Note: all input parameters are detached for fast computation
        Return -S in dense torch.tensor representation
    :param opt_vars: gamma, lbdas, nus, etas, and s
    :param weights: Ws, and bs
    :param x0: raw input
    :param y0: correct label
    :param yp: adversarial label
    :param eps: perturbation level
    :return: the matrix
    """
    gammas, lbdas, nus, etas, s = opt_vars
    Ws, bs = weights

    if numerical:
        gammas = [x.cpu().detach() for x in gammas]
        lbdas = [x.cpu().detach() for x in lbdas]
        nus = [x.cpu().detach() for x in nus]
        etas = [x.cpu().detach() for x in etas]
        s = s.cpu().detach()

        Ws = [x.cpu().detach() for x in Ws]
        bs = [x.cpu().detach() for x in bs]

        x0, y0 = x0.cpu(), y0.cpu()

    if numerical:
        torch.set_grad_enabled(False)

    last_ele = - s \
               - torch.sum(torch.stack([torch.dot(x, y) for (x, y) in zip(nus, bs)])) * 2.0 \
               - torch.sum(torch.stack([torch.dot(x, torch.tensor(y * z).cuda() if x.is_cuda else torch.tensor(y * z)) for (x, y, z) in zip(gammas, l, u)])) * 2.0 \
               + (- bs[-1][y0] + bs[-1][yp]) * 2.0
    if numerical:
        last_ele_t = last_ele.view(1)
    else:
        last_ele_t = last_ele.view(1).cuda()
    diag = [- s - 2.0 * gammas[i] - 2.0 * (lbdas[i-1] if i > 0 else 0.0) for i in range(len(gammas))] + [last_ele_t]
    diag = torch.cat(diag)
    mat = torch.diag(diag)

    dim_idx = 0
    for i in range(len(Ws) - 1):
        ni1, ni = Ws[i].size(0), Ws[i].size(1)

        il, ir = dim_idx + ni, dim_idx + ni + ni1
        jl, jr = dim_idx, dim_idx + ni
        local_mat = (Ws[i].t() * lbdas[i]).t()
        mat[il: ir, jl: jr] = local_mat

        il, ir = dim_idx, dim_idx + ni
        jl, jr = dim_idx + ni, dim_idx + ni + ni1
        mat[il: ir, jl: jr] = local_mat.t()

        dim_idx += ni
    dim_idx += Ws[-1].size(1)

    dim_idx_j = 0
    for i in range(len(Ws)):
        if numerical:
            local_vec = torch.zeros(Ws[i].size(1), dtype=torch.double)
        else:
            local_vec = torch.zeros(Ws[i].size(1), dtype=torch.double).cuda()
        local_vec += gammas[i] * (torch.tensor(l[i] + u[i]).cuda() if gammas[i].is_cuda else torch.tensor(l[i] + u[i]))
        if i < len(Ws) - 1:
            local_vec -= torch.matmul(Ws[i].t(), nus[i])
        else:
            local_vec += - Ws[i][y0] + Ws[i][yp]
        if i > 0:
            local_vec += nus[i-1] + etas[i-1] + lbdas[i-1] * bs[i-1]
        mat[dim_idx, dim_idx_j: dim_idx_j + Ws[i].size(1)] = local_vec
        mat[dim_idx_j: dim_idx_j + Ws[i].size(1), dim_idx] = local_vec
        dim_idx_j += Ws[i].size(1)

    mat = mat.contiguous()

    if numerical:
        torch.set_grad_enabled(True)

    return mat


def max_eigenvalue(mat: torch.Tensor, eps=1e-6, dup=5):
    n = mat.size(0)
    tot = 0.
    tim = 0
    with torch.no_grad():
        for i in range(dup):
            now = None
            v = torch.randn(n, dtype=torch.double)
            while True:
                tim += 1
                Av = torch.matmul(mat, v)
                vlen = torch.norm(Av)
                tmp = vlen / torch.norm(v)
                if now is None or abs(tmp - now) / now > eps:
                    now = tmp
                else:
                    now = tmp
                    break
                v = Av * (1.0 / vlen)
            tot += now.item()
    return tot / dup


class CompressMatrix():
    def __init__(self, opt_vars, weights, x0, y0, yp, l, u, shapes):
        """
            Note: all input parameters are detached for fast computation
            Calculate -S in latent sparse representation
            :param opt_vars: gamma, lbdas, nus, etas, and s
            :param weights: Ws, and bs
            :param x0: raw input
            :param y0: correct label
            :param yp: adversarial label
            :param eps: perturbation level
            :return: the matrix
        """
        gammas, lbdas, nus, etas, s = opt_vars
        Ws, bs = weights

        gammas = [x.cpu().detach() for x in gammas]
        lbdas = [x.cpu().detach() for x in lbdas]
        nus = [x.cpu().detach() for x in nus]
        etas = [x.cpu().detach() for x in etas]
        s = s.cpu().detach()

        Ws = [x.cpu().detach() for x in Ws]
        bs = [x.cpu().detach() for x in bs]

        x0, y0 = x0.cpu(), y0.cpu()

        block_idxs = list()
        for i in range(len(Ws)):
            if i == 0:
                block_idxs.append(Ws[i].size()[1])
            else:
                block_idxs.append(block_idxs[-1] + Ws[i-1].size()[0])

        with torch.no_grad():
            tri_diag = [- s - 2.0 * gammas[i] - 2.0 * (lbdas[i-1] if i > 0 else 0.0) for i in range(len(gammas))]
            tri_diag = torch.cat(tri_diag)

            l_mats = list()
            for i in range(len(Ws) - 1):
                local_mat = (Ws[i].t() * lbdas[i]).t()
                l_mats.append(local_mat)

            last_diag = - s \
                        - torch.sum(torch.stack([torch.dot(x, y) for (x, y) in zip(nus, bs)])) * 2.0 \
                        - torch.sum(torch.stack([torch.dot(x, torch.tensor(y * z).cuda() if x.is_cuda else torch.tensor(y * z)) for (x, y, z) in zip(gammas, l, u)])) * 2.0 \
                        + (- bs[-1][y0] + bs[-1][yp]) * 2.0

            last_vec = list()
            for i in range(len(Ws)):
                local_vec = torch.zeros(Ws[i].size(1), dtype=torch.double)
                local_vec += gammas[i] * (torch.tensor(l[i] + u[i]).cuda() if gammas[i].is_cuda else torch.tensor(l[i] + u[i]))
                if i < len(Ws) - 1:
                    local_vec -= torch.matmul(Ws[i].t(), nus[i])
                else:
                    local_vec += - Ws[i][y0] + Ws[i][yp]
                if i > 0:
                    local_vec += nus[i-1] + etas[i-1] + lbdas[i-1] * bs[i-1]
                last_vec.append(local_vec)
            last_vec = torch.cat(last_vec)

        self.block_idxs = block_idxs
        self.n = self.block_idxs[-1] + 1
        self.tri_diag = tri_diag
        self.l_mats = l_mats
        self.last_diag = last_diag
        self.last_vec = last_vec

        self.max_eigen = None
        self.subL = None
        self.subU = None
        self.subUinv = None
        self.inv = None

    def mul_vec(self, x):
        with torch.no_grad():
            ans = torch.zeros(self.n, dtype=torch.double)
            ans[:-1] += self.tri_diag * x[:-1] + self.last_vec * x[-1]
            ans[-1] = torch.dot(self.last_vec, x[:-1]) + self.last_diag * x[-1]
            for i in range(len(self.l_mats)):
                xl = self.block_idxs[i - 1] if i > 0 else 0
                ans[self.block_idxs[i]: self.block_idxs[i + 1]] += torch.matmul(self.l_mats[i], x[xl: self.block_idxs[i]])
                ans[xl: self.block_idxs[i]] += torch.matmul(self.l_mats[i].t(), x[self.block_idxs[i]: self.block_idxs[i + 1]])
        return ans

    def max_eigenvalue(self, eps=1e-6, dup=5):
        n = self.n
        tot = 0.
        tim = 0
        with torch.no_grad():
            for i in range(dup):
                now = None
                v = torch.randn(n, dtype=torch.double)
                while True:
                    tim += 1
                    Av = self.mul_vec(v)
                    vlen = torch.norm(Av)
                    tmp = vlen / torch.norm(v)
                    if now is None or abs(tmp - now) / now > eps:
                        now = tmp
                    else:
                        now = tmp
                        break
                    v = Av * (1.0 / vlen)
                tot += now.item()
        self.max_eigen = tot / dup
        return self.max_eigen

    def LU_decomposition(self):
        with torch.no_grad():
            U = [torch.diag(self.tri_diag[:self.block_idxs[0]])]
            U_inv = [torch.inverse(x) for x in U]
            L = list()
            for i in range(1, len(self.block_idxs)):
                L.append(torch.matmul(self.l_mats[i-1], U_inv[i-1]))
                U.append(torch.diag(self.tri_diag[self.block_idxs[i-1]: self.block_idxs[i]]) -
                         torch.matmul(L[-1], self.l_mats[i-1].t()))
                U_inv.append(torch.inverse(U[-1]))
        self.subL = L
        self.subU = U
        self.subUinv = U_inv

    def nicer_LU(self):
        """
            Recover full torch tensor of L and U matrix of upper-left submatrix
            Just for print and test use
        :return: L and U matrix
        """
        assert self.subL is not None and self.subU is not None
        matL = torch.eye(self.n - 1)
        matU = torch.zeros(self.n - 1, self.n - 1)
        for i in range(len(self.subL)):
            sl = self.block_idxs[i - 1] if i > 0 else 0
            matL[self.block_idxs[i]: self.block_idxs[i + 1], sl: self.block_idxs[i]] = self.subL[i]
        for i in range(len(self.l_mats)):
            sl = self.block_idxs[i - 1] if i > 0 else 0
            matU[sl: self.block_idxs[i], self.block_idxs[i]: self.block_idxs[i + 1]] = self.l_mats[i].t()
        for i in range(len(self.subU)):
            sl = self.block_idxs[i - 1] if i > 0 else 0
            matU[sl: self.block_idxs[i], sl: self.block_idxs[i]] = self.subU[i]
        return matL, matU

    def tri_submatrix_inverse(self):
        with torch.no_grad():
            L_inv = torch.eye(self.n - 1)
            for i in range(len(self.subL)):
                ll, l, r = (self.block_idxs[i - 1] if i > 0 else 0), self.block_idxs[i], self.block_idxs[i + 1]
                L_inv[l: r] = L_inv[l: r] - torch.matmul(self.subL[i], L_inv[ll: l])

            ans = L_inv
            for i in range(len(self.subU) - 1, -1, -1):
                sl = self.block_idxs[i - 1] if i > 0 else 0
                if i < len(self.subU) - 1:
                    ans[sl: self.block_idxs[i]] -= torch.matmul(self.l_mats[i].t(), ans[self.block_idxs[i]: self.block_idxs[i + 1]])
                ans[sl: self.block_idxs[i]] = torch.matmul(self.subUinv[i], ans[sl: self.block_idxs[i]])

        return ans

    def inverse(self):
        """
            fast inverse computation making use of matrix structure
        :return:
        """
        with torch.no_grad():
            # t0 = time.time()
            self.LU_decomposition()
            # t1 = time.time()
            submat_inv = self.tri_submatrix_inverse()
            # t2 = time.time()
            reci = 1.0 / \
                   (self.last_diag - torch.chain_matmul(self.last_vec.view(1, -1), submat_inv, self.last_vec.view(-1, 1)).item())
            inv_vec = - reci * torch.matmul(submat_inv, self.last_vec)

            ans = torch.zeros(self.n, self.n)
            ans[:-1, :-1] = submat_inv + \
                            reci * torch.chain_matmul(submat_inv, self.last_vec.view(-1, 1),
                                                      self.last_vec.view(1, -1), submat_inv)
            ans[-1, :-1] = inv_vec
            ans[:-1, -1] = inv_vec
            ans[-1, -1] = reci
            # t3 = time.time()

        # print(t1 - t0, '+', t2 - t1, '+', t3 - t2)

        self.inv = ans
        return ans



