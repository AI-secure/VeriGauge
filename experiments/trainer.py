GPU_ID = 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
import getpass

from sys import stderr
from pprint import pprint
import time
import datetime
import json

import numpy as np
import numpy.linalg as la

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

from convex_adversarial.convex_adversarial.dual_network import robust_loss

from models.exp_model import SAVE_PATH, try_load_weight
from experiments.params import *
from datasets import get_dataset, get_input_shape
from models.test_model import _IMAGENET_MEAN, _IMAGENET_STDDEV, _CIFAR10_MEAN, _CIFAR10_STDDEV, _MNIST_MEAN, _MNIST_STDDEV
import model
from models.test_model import NormalizeLayer

# ======== GLOBAL PARAMS ========
DEBUG = False
# DEBUG = True
LR_REDUCE = 10
LR_REDUCE_RATE = 0.2

EPS_WARMUP_EPOCHS = 1
# verbose step
STEP = 10

def calc_tot_torun():
    ans = 0
    for (ds, model_name, _) in model_order:
        for now_method in method_order:
            if now_method == 'clean':
                ans += 1
            else:
                ans += len(eps_list[ds])
    return ans

def main(train_method, dataset, model_name, params):
    # prepare dataset and normalize settings
    normalize = None
    if params.get('normalized', False):
        if dataset == 'mnist':
            normalize = (_MNIST_MEAN, _MNIST_STDDEV)
        elif dataset == 'cifar10':
            normalize = (_CIFAR10_MEAN, _CIFAR10_STDDEV)
        elif dataset == 'imagenet':
            normalize = (_IMAGENET_MEAN, _IMAGENET_STDDEV)
    train_set = get_dataset(dataset, 'train', normalize)
    test_set = get_dataset(dataset, 'test', normalize)

    # read input shape (c, h, w)
    input_shape = get_input_shape(dataset)

    # read params
    batch_size = params['batch_size']
    optimizer_name = params.get('optimizer', 'sgd')
    if optimizer_name == 'sgd':
        lr = params.get('learning_rate', 0.1)
        momentum = params.get('momentum', 0.1)
        weight_decay = params.get('weight_decay', 5e-4)
    elif optimizer_name == 'adam':
        lr = params.get('learning_rate', 0.1)
    else:
        raise NotImplementedError
    cur_lr = lr
    print('default learning rate =', cur_lr, file=stderr)
    start_epoch = 0
    epochs = params.get('epochs', 0)
    eps = normed_eps = params['eps']
    if train_method == 'adv':
        # Note: for adversarial training, in training phase, we use the manual implementation version for precision,
        # and use the clearhans implementation in test phase for precision
        eps_iter_coef = params['eps_iter_coef']
        clip_min = params['clip_min']
        clip_max = params['clip_max']
        if normalize is not None:
            mean, std = normalize
            clip_min = (clip_min - max(mean)) / min(std) - 1e-6
            clip_max = (clip_max - min(mean)) / min(std) + 1e-6
            normed_eps = eps / min(std)
        nb_iter = params['nb_iter']
        rand_init = params['rand_init']

        adv_params = {'eps': normed_eps,
                      'clip_min': clip_min,
                      'clip_max': clip_max,
                      'eps_iter': eps_iter_coef * eps,
                      'nb_iter': nb_iter,
                      'rand_init': rand_init}
    elif train_method == 'certadv':
        # Note: for certified adversarially trained models, we test its accuracy still using PGD attack
        eps_iter_coef = params['eps_iter_coef']
        clip_min = params['clip_min']
        clip_max = params['clip_max']
        if normalize is not None:
            mean, std = normalize
            clip_min = (clip_min - max(mean)) / min(std) - 1e-6
            clip_max = (clip_max - min(mean)) / min(std) + 1e-6
            normed_eps = eps / min(std)
        nb_iter = params['nb_iter']
        rand_init = params['rand_init']

        adv_params = {'eps': normed_eps,
                      'clip_min': clip_min,
                      'clip_max': clip_max,
                      'eps_iter': eps_iter_coef * eps,
                      'nb_iter': nb_iter,
                      'rand_init': rand_init}
        print(adv_params, file=stderr)

    # prepare loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, pin_memory=True)

    # stats
    train_tot = len(train_set)
    test_tot = len(test_set)

    best_acc = 0.0
    best_robacc = 0.0

    # load model
    m = model.load_model('exp', dataset, model_name).cuda()
    print(m)

    if train_method == 'adv' and params['retrain']:
        # retrain from the best clean model
        clean_model_name = f'{dataset}_{model_name}_clean_0_best'
        new_m, stats = try_load_weight(m, clean_model_name)
        assert stats == True, "Could not load pretrained clean model."
        if isinstance(new_m[0], NormalizeLayer):
            # squeeze the normalize layer out
            new_m = new_m[1]
        m = new_m
    elif train_method == 'certadv':
        configdir = params['configpath']
        ds_mapping = {'cifar10': 'cifar', 'mnist': 'mnist'}
        ds_multiplier = {'cifar10': 255., 'mnist': 10.}
        configfilename = f'exp_{ds_mapping[dataset]}{int(round(eps * ds_multiplier[dataset]))}.json'
        with open(os.path.join(configdir, configfilename), 'r') as f:
            real_config = json.load(f)
        epochs = real_config['training_params']['epochs']
        start_epoch = epochs - 1
        model_path = os.path.join(os.path.join(real_config['path_prefix'], real_config['models_path']), f'{model_name}_best.pth')
        d = torch.load(model_path)
        print(f'certadv load from {model_path}', file=stderr)
        m.load_state_dict(d['state_dict'])


    # open file handler
    save_name = f'{ds}_{model_name}_{now_method}_{eps}'
    mode = 'a'
    if os.path.exists(f'{SAVE_PATH}/{save_name}_train.log') or os.path.exists(f'{SAVE_PATH}/{save_name}_test.log'):
        choice = getpass.getpass(
            f'Log exists. Do you want to rewrite it? (Y/others) ')
        if choice == 'Y':
            mode = 'w'
            print('Rewrite log', file=stderr)
        else:
            mode = 'a'
    train_log = open(f'{SAVE_PATH}/{save_name}_train.log', mode)
    test_log = open(f'{SAVE_PATH}/{save_name}_test.log', mode)

    # special treatment for model G - layerwise training
    if model_name == 'G' and train_method == 'adv':
        new_last_layer = nn.Linear(1024, 10)

    # start
    for epoch in range(start_epoch, epochs):

        if epoch % LR_REDUCE == 0 and epoch > 0:
            # learning rate reduced to LR_REDUCE_RATE every LR_REDUCE epochs
            cur_lr *= LR_REDUCE_RATE
            print(f'  reduce learning rate to {cur_lr}', file=stderr)

        # special treatment for model G - layerwise training
        if model_name == 'G' and train_method == 'adv':
            new_m = list()
            tmp_cnt = 0
            for l in m:
                new_m.append(l)
                if isinstance(l, nn.Linear) and l.out_features == 1024:
                    tmp_cnt += 1
                if tmp_cnt > epoch / 5:
                    if l.out_features == 1024:
                        new_m.append(nn.ReLU())
                        new_m.append(new_last_layer)
                    break
            new_m = nn.Sequential(*new_m).cuda()
            m, new_m = new_m, m
            print(m, file=stderr)
            cur_lr = lr
            print(f'  learning rate restored to {cur_lr}', file=stderr)

        # init optimizer
        if optimizer_name == 'adam':
            opt = optim.Adam(m.parameters(), lr=cur_lr)
        elif optimizer_name == 'sgd':
            opt = optim.SGD(m.parameters(), lr=cur_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise Exception("Fail to create the optimizer")

        cur_idx = 0
        cur_acc = 0.0
        cur_robacc = 0.0

        batch_tot = 0
        batch_acc_tot = 0
        batch_robacc_tot = 0

        clean_ce = 0.0
        adv_ce = 0.0

        # now eps
        now_eps = normed_eps * min((epoch + 1) / EPS_WARMUP_EPOCHS, 1.0)
        # =========== Training ===========
        print(f'Epoch {epoch}: training', file=stderr)
        if train_method != 'clean':
            print(f'  Training eps={now_eps:.3f}', file=stderr)
        m.train()

        for i, (X,y) in enumerate(train_loader):

            if DEBUG and i > 10:
                break

            start_t = time.time()

            X_clean, y_clean = X.cuda(), y.cuda().long()
            clean_out = m(Variable(X_clean))
            clean_ce = nn.CrossEntropyLoss()(clean_out, Variable(y_clean))

            batch_tot = X.size(0)
            batch_acc_tot = (clean_out.data.max(1)[1] == y_clean).float().sum().item()

            if train_method == 'clean':
                opt.zero_grad()
                clean_ce.backward()
                opt.step()

            elif train_method == 'adv':
                X_pgd = Variable(X, requires_grad=True)
                for _ in range(nb_iter):
                    opt_pgd = optim.Adam([X_pgd], lr=1e-3)
                    opt.zero_grad()
                    loss = nn.CrossEntropyLoss()(m(X_pgd.cuda()), Variable(y_clean))
                    loss.backward()
                    eta = now_eps * eps_iter_coef * X_pgd.grad.data.sign()
                    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                    eta = torch.clamp(X_pgd.data - X, -now_eps, now_eps)
                    X_pgd.data = X + eta
                    X_pgd.data = torch.clamp(X_pgd.data, clip_min, clip_max)

                # print(X_pgd.data, la.norm((X_pgd.data - X).numpy().reshape(-1), np.inf), file=stderr)
                adv_out = m(Variable(X_pgd.data).cuda())
                adv_ce = nn.CrossEntropyLoss()(adv_out, Variable(y_clean))
                batch_robacc_tot = (adv_out.data.max(1)[1] == y_clean).float().sum()

                opt.zero_grad()
                adv_ce.backward()
                opt.step()

            elif train_method == 'certadv':
                # no action to do for training
                adv_ce = torch.Tensor([0.0]).cuda()
                pass

            end_t = time.time()

            clean_ce = clean_ce.detach().cpu().item()
            if train_method != 'clean':
                adv_ce = adv_ce.detach().cpu().item()

            runtime = end_t - start_t
            cur_acc = (cur_acc * cur_idx + batch_acc_tot) / (cur_idx + batch_tot)
            if train_method != 'clean':
                cur_robacc = (cur_robacc * cur_idx + batch_robacc_tot) / (cur_idx + batch_tot)
            cur_idx += batch_tot

            print(f'{epoch} {cur_idx} {cur_acc} {cur_robacc} {batch_acc_tot/batch_tot:.3f} {batch_robacc_tot/batch_tot:.3f}'
                  f' {clean_ce:.3f} {adv_ce:.3f} {runtime:.3f}', file=train_log)
            if i % STEP == 0 or cur_idx == train_tot:
                print(f'  [train] {epoch}/{cur_idx} acc={cur_acc:.3f}({batch_acc_tot/batch_tot:.3f}) '
                      f'robacc={cur_robacc:.3f}({batch_robacc_tot/batch_tot:.3f}) ce={clean_ce:.3f} adv_ce={adv_ce:.3f} time={runtime:.3f}', file=stderr)


        train_log.flush()

        # =========== Testing ===========
        print(f'Epoch {epoch}: testing', file=stderr)
        m.eval()
        torch.set_grad_enabled(False)

        cur_idx = 0
        cur_acc = 0.0
        cur_robacc = 0.0

        batch_tot = 0
        batch_acc_tot = 0
        batch_robacc_tot = 0

        clean_ce = 0.0
        adv_ce = 0.0

        if train_method in ['adv', 'certadv']:
            tf_model = convert_pytorch_model_to_tf(m)
            ch_model = CallableModelWrapper(tf_model, output_layer='logits')
            x_op = tf.placeholder(tf.float32, shape=(None,) + tuple(input_shape))
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))
            attk = ProjectedGradientDescent(ch_model, sess=sess)
            adv_x = attk.generate(x_op, **adv_params)
            adv_preds_op = tf_model(adv_x)

        for i, (X,y) in enumerate(test_loader):

            if DEBUG and i >= 10:
                break

            start_t = time.time()

            X_clean, y_clean = X.cuda(), y.cuda().long()
            clean_out = m(Variable(X_clean))
            clean_ce = nn.CrossEntropyLoss()(clean_out, Variable(y_clean))

            batch_tot = X.size(0)
            batch_acc_tot = (clean_out.data.max(1)[1] == y_clean).float().sum().item()

            if train_method in ['adv', 'certadv']:

                (adv_preds,) = sess.run((adv_preds_op,),
                                        feed_dict={x_op: X})
                adv_preds = torch.Tensor(adv_preds)

                adv_ce = nn.CrossEntropyLoss()(adv_preds, Variable(y))
                batch_robacc_tot = (adv_preds.data.max(1)[1] == y).float().sum().item()

            # elif train_method == 'certadv':
            #
            #     adv_ce, robust_err = robust_loss(m, eps,
            #                                      Variable(X_clean), Variable(y_clean),
            #                                      proj=50, norm_type='l1_median', bounded_input=True)
            #
            #     batch_robacc_tot = (1.0 - robust_err) * batch_tot

            end_t = time.time()

            clean_ce = clean_ce.detach().cpu().item()
            if train_method != 'clean':
                adv_ce = adv_ce.detach().cpu().item()

            runtime = end_t - start_t
            cur_acc = (cur_acc * cur_idx + batch_acc_tot) / (cur_idx + batch_tot)
            if train_method != 'clean':
                cur_robacc = (cur_robacc * cur_idx + batch_robacc_tot) / (cur_idx + batch_tot)
            cur_idx += batch_tot

            print(
                f'{epoch} {cur_idx} {cur_acc} {cur_robacc} {batch_acc_tot / batch_tot:.3f} {batch_robacc_tot / batch_tot:.3f}'
                f' {clean_ce} {adv_ce} {runtime:.3f}', file=test_log)
            if i % STEP == 0 or cur_idx == train_tot:
                print(f'  [test] {epoch}/{cur_idx} acc={cur_acc:.3f}({batch_acc_tot / batch_tot:.3f}) '
                      f'robacc={cur_robacc:.3f}({batch_robacc_tot / batch_tot:.3f}) time={runtime:.3f}', file=stderr)

        torch.set_grad_enabled(True)

        if model_name == 'G' and train_method == 'adv':
            # switch back
            m, new_m = new_m, m

        def save_with_configs(m, path):
            torch.save({
                'state_dict': m.state_dict(),
                'acc': cur_acc,
                'robacc': cur_robacc,
                'epoch': epoch,
                'normalized': normalize is not None,
                'dataset': dataset
            }, path)

        if not os.path.exists(f'{SAVE_PATH}/{save_name}_chkpt'):
            os.makedirs(f'{SAVE_PATH}/{save_name}_chkpt')
        save_with_configs(m, f'{SAVE_PATH}/{save_name}_chkpt/{save_name}_ep_{epoch:03d}.pth')
        if (train_method == 'clean' and cur_acc > best_acc) or (train_method != 'clean' and cur_robacc > best_robacc):
            save_with_configs(m, f'{SAVE_PATH}/{save_name}_best.pth')
            print(f"  Updated, acc {best_acc:.3f} => {cur_acc:.3f} robacc {best_robacc:.3f} => {cur_robacc:.3f}", file=stderr)
            best_acc = cur_acc
            best_robacc = cur_robacc

        test_log.flush()

        # memory clean after each batch
        torch.cuda.empty_cache()
        if train_method == 'adv':
            sess.close()

    train_log.close()
    test_log.close()

if __name__ == '__main__':

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f'crease folder {SAVE_PATH}', file=stderr)

    tot_torun = calc_tot_torun()
    run_counter = 0
    for now_method in method_order:
        for (ds, model_name, batch_size) in model_order:
            if now_method == 'clean':
                params = clean_params.copy()
            elif now_method == 'adv':
                params = adv_params.copy()
            elif now_method == 'certadv':
                params = certadv_params.copy()
            if now_method in ['adv', 'certadv']:
                now_epslist = eps_list[ds].copy()
            else:
                now_epslist = [0]
            batch_size = int(batch_size * batch_size_multipler[now_method])
            for now_eps in now_epslist:
                run_counter += 1

                params['eps'] = now_eps
                params['batch_size'] = batch_size

                print(f'[{run_counter}/{tot_torun}]', 'method:', now_method, '|', 'dataset:', ds, '|', 'model:', model_name, file=stderr)
                pprint(params, stream=stderr)

                save_name = f'{ds}_{model_name}_{now_method}_{now_eps}'
                cond = True
                if os.path.exists(f'{SAVE_PATH}/{save_name}_best.pth'):
                    try:
                        d = torch.load(f'{SAVE_PATH}/{save_name}_best.pth')
                        acc = d['acc']
                        robacc = d['robacc']
                    except:
                        acc, robacc = 0.0, 0.0
                    choice = getpass.getpass(f'The weight {save_name} (acc={acc:.3f}, robacc={robacc:.3f}) already exists. Do you want to retrain it? (Y/others) ')
                    if choice == 'Y':
                        cond = True
                        print('Retrain selected', file=stderr)
                    else:
                        cond = False
                        print('Skip selected', file=stderr)
                if not cond:
                    continue

                main(now_method, ds, model_name, params)

