
from model import model_list, normalization, input_shape
from model import mnist_model_toy, cifar_model_toy, mnist_model_tiny, cifar_model_tiny
from utils import data_loaders


import argparse
import os
import numpy as np

import torch
import torch.optim as optim
from torch import nn

name_mapping = {
    ('mnist', 'toy', 'random', None): 'mnist.toy.random',
    ('mnist', 'toy', 'clean', None): 'mnist.toy.clean',
    ('mnist', 'toy', 'pgdadv', '0.1'): 'mnist.toy.pgdadv.0.1',
    ('mnist', 'toy', 'pgdadv', '0.3'): 'mnist.toy.pgdadv.0.3',
    ('cifar', 'toy', 'random', None): 'cifar.toy.random',
    ('cifar', 'toy', 'clean', None): 'cifar.toy.clean',
    ('cifar', 'toy', 'pgdadv', '0.0349'): 'cifar.toy.pgdadv.2',
    ('cifar', 'toy', 'pgdadv', '0.1394'): 'cifar.toy.pgdadv.8',
    ('mnist', 'tiny', 'random', None): 'mnist.tiny.random',
    ('mnist', 'tiny', 'clean', None): 'mnist.tiny.clean',
    ('cifar', 'tiny', 'random', None): 'cifar.tiny.random',
    ('cifar', 'tiny', 'clean', None): 'cifar.tiny.clean'
}


def load_model(dataset, model_size):
    if model_size == 'toy':
        if dataset == 'mnist':
            return mnist_model_toy()
        elif dataset == 'cifar':
            return cifar_model_toy()
    elif model_size == 'tiny':
        if dataset == 'mnist':
            return mnist_model_tiny()
        elif dataset == 'cifar':
            return cifar_model_tiny()
    raise NotImplementedError()


"""
    adapted from https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py
"""
def to_var(x, requires_grad=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.tensor(x, requires_grad=requires_grad)


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, random_start=True):
        """
            Attack parameter initialization. The attack performs k steps of
            size a, while always staying within epsilon from the initial
            point.
            https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py

            NOTE: the epsilon has not been rescaled
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, in_min=0., in_max=1.):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, in_min, in_max) # ensure valid pixel range

        return X


parser = argparse.ArgumentParser(description='Calculate model adversarial lower bound by PGD')
parser.add_argument('--cifar', default=False, action='store_true')
parser.add_argument('--mnist', default=False, action='store_true')
parser.add_argument('--toy', default=False, action='store_true')
parser.add_argument('--tiny', default=False, action='store_true')
parser.add_argument('--random', default=False, action='store_true')
parser.add_argument('--clean', default=False, action='store_true')
parser.add_argument('--pgdadv', default=False, action='store_true')
parser.add_argument('--eps')
parser.add_argument('--cuda_ids', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--pgd_step', type=int, default=20)
if __name__ == '__main__':
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_ids)

    tasklist = list()
    for dataset in ['mnist', 'cifar']:
        cont = (dataset == 'mnist' and args.mnist) or (dataset == 'cifar' and args.cifar)
        if cont:
            for model_size in ['toy', 'tiny']:
                cont = (model_size == 'toy' and args.toy) or (model_size == 'tiny' and args.tiny)
                if cont:
                    for mode in ['random', 'clean', 'pgdadv']:
                        cont = (mode == 'random' and args.random) or (mode == 'clean' and args.clean) or (mode == 'pgdadv' and args.pgdadv and args.eps is not None)
                        if cont:
                            tasklist.append((dataset, model_size, mode, args.eps if mode == 'pgdadv' else None))

    for task in tasklist:
        dataset, model_size, mode, eps = task
        model = load_model(dataset, model_size)

        if dataset == 'mnist':
            mean = normalization['mnist'][0]
            std = [normalization['mnist'][1] for _ in mean]
        elif dataset == 'cifar':
            mean = normalization['cifar'][0]
            std = [normalization['cifar'][1] for _ in mean]
        in_min, in_max = (0.0 - max(mean)) / min(std), (1.0 - min(mean)) / max(std)
        train_loader, test_loader = data_loaders(dataset, args.batch_size,
                                                 shuffle_test=False, norm_mean=mean, norm_std=std)

        if mode == 'random':
            # direct save :)
            model = model.cuda()
            torch.save(model.state_dict(), model_list[name_mapping[task]][2])
        elif mode == 'clean' or mode == 'pgdadv':
            # direct train :)
            # pgd train :)

            model = model.cuda()
            now_lr = args.lr

            step = 10

            if mode == 'pgdadv':
                attacker = LinfPGDAttack()
                eps = float(args.eps)

            for ep in range(args.epochs):
                print(f'epoch {ep}')
                if ep % 10 == 0 and ep > 0:
                    # slow it down a bit
                    now_lr /= 3.
                opt = optim.SGD(model.parameters(), lr=now_lr, momentum=args.momentum, weight_decay=args.weight_decay)
                tot, tot_correct, now_step = 0, 0, 0
                for X, y in train_loader:
                    if mode == 'clean':
                        pass
                    else:
                        # apply attack
                        X, y = X.numpy(), y.numpy()
                        attacker = LinfPGDAttack(model, eps, k=args.pgd_step, a=eps / args.pgd_step * 2.)
                        X = attacker.perturb(X, y, in_min, in_max)
                        X, y = torch.tensor(X), torch.tensor(y)
                    X, y = X.cuda(), y.cuda()

                    out = model(X)
                    err = nn.CrossEntropyLoss()(out, y)

                    loss = err.item()
                    acc = ((torch.argmax(out, dim=1)) == y).sum().item()

                    opt.zero_grad()
                    err.backward()
                    opt.step()

                    tot += X.shape[0]
                    tot_correct += acc
                    now_step += 1
                    if now_step >= step:
                        now_step = 0
                        print(f'  epoch {ep}, samples {tot}, loss {loss:.3}, now acc {acc / X.shape[0]:.3}, tot acc {tot_correct / tot:.3}')

                print('test set')
                tot, tot_correct, now_step = 0, 0, 0
                best = 0.0
                for X, y in test_loader:
                    if mode == 'clean':
                        pass
                    else:
                        # apply attack
                        X, y = X.numpy(), y.numpy()
                        attacker = LinfPGDAttack(model, eps, k=args.pgd_step, a=eps / args.pgd_step * 2.)
                        X = attacker.perturb(X, y, in_min, in_max)
                        X, y = torch.tensor(X), torch.tensor(y)
                    X, y = X.cuda(), y.cuda()

                    with torch.no_grad():
                        out = model(X)
                        acc = ((torch.argmax(out, dim=1)) == y).sum().item()

                    tot += X.shape[0]
                    tot_correct += acc
                    now_step += 1
                    if now_step >= step:
                        now_step = 0
                        print(f'  epoch {ep}, samples {tot}, tot acc {tot_correct / tot:.3}')
                if tot_correct / tot > best:
                    print(f'save with {tot_correct / tot:.3}')
                    torch.save(model.state_dict(), model_list[name_mapping[task]][2])

        print(f'{task} done')
