import sys
sys.path.append('.')

import argparse

import os
import time
import setproctitle

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Variable

import cleverhans
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

from model import load_model, normalization, input_shape
from utils import data_loaders

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser(description='Calculate model adversarial lower bound by PGD')
parser.add_argument('--dataset')
parser.add_argument('--eps', type=float)
parser.add_argument('--model')
parser.add_argument('--attack')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda_ids', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_ids)

    # load model
    model = load_model(args.model)

    eps = args.eps
    if args.dataset == 'mnist':
        mean = normalization['mnist'][0]
        std = [normalization['mnist'][1] for _ in mean]
        x = input_shape['mnist']
        x_op = tf.placeholder(tf.float32, shape=(None, x[0], x[1], x[2],))
    elif args.dataset == 'cifar':
        mean = normalization['cifar'][0]
        std = [normalization['cifar'][1] for _ in mean]
        x = input_shape['cifar']
        x_op = tf.placeholder(tf.float32, shape=(None, x[0], x[1], x[2],))
    train_loader, test_loader = data_loaders(args.dataset, args.batch_size,
                                             shuffle_test=False, norm_mean=mean, norm_std=std)

    sess = tf.Session(config=config)

    tf_model = convert_pytorch_model_to_tf(model)
    ch_model = CallableModelWrapper(tf_model, output_layer='logits')

    if args.attack == 'CW':
        attk = CarliniWagnerL2(ch_model, sess=sess)
        params = {'binary_search_steps': 10,
                  'max_iterations': 100,
                  'learning_rate': 0.2,
                  'batch_size': args.batch_size,
                  'initial_const': 10}
    elif args.attack == 'PGD':
        attk = ProjectedGradientDescent(ch_model, sess=sess)
        clip_min = (0.0 - 1E-6 - max(mean)) / std[0]
        clip_max = (1.0 + 1E-6 - min(mean)) / std[0]
        params = {'eps': eps,
                  'clip_min': clip_min,
                  'clip_max': clip_max,
                  'eps_iter': 0.005,
                  'nb_iter': 100,
                  'rand_init': False}
    elif args.attack == 'FGSM':
        attk = FastGradientMethod(ch_model, sess=sess)
        clip_min = (0.0 - 1E-6 - max(mean)) / std[0]
        clip_max = (1.0 + 1E-6 - min(mean)) / std[0]
        params = {'eps': eps,
                  'clip_min': clip_min,
                  'clip_max': clip_max}

    adv_x = attk.generate(x_op, **params)
    adv_preds_op = tf_model(adv_x)

    stime = time.time()

    tot_clean_err, tot_adv_err, tot = 0.0, 0.0, 0
    # tot_adv_input_err = 0.0

    clean_detail = list()
    detail = list()

    for i, (xs, ys) in enumerate(test_loader):
        ys = ys.numpy()

        clean_preds = model(xs.cuda()).detach().cpu().numpy()

        # adv_input = sess.run(adv_x, feed_dict={x_op: xs})
        # adv_input_preds = model(torch.Tensor(adv_input).cuda()).detach().cpu().numpy()
        (adv_preds,) = sess.run((adv_preds_op,), feed_dict={x_op: xs})

        clean_err = float((np.argmax(clean_preds, axis=1) != ys).sum()) / xs.size(0)
        adv_err = float(((np.argmax(adv_preds, axis=1) != ys) + (np.argmax(clean_preds, axis=1) != ys)).sum()) / xs.size(0)
        # adv_input_err = float((np.argmax(adv_input_preds, axis=1) != ys).sum()) / xs.size(0)

        clean_detail.extend((np.argmax(clean_preds, axis=1) != ys))
        detail.extend(((np.argmax(adv_preds, axis=1) != ys) + (np.argmax(clean_preds, axis=1) != ys)).tolist())

        tot += xs.size(0)
        tot_clean_err += clean_err * xs.size(0)
        tot_adv_err += adv_err * xs.size(0)
        # tot_adv_input_err += adv_input_err * xs.size(0)

        # print(i, clean_err, adv_err, adv_input_err)
        print('index: {0}, tot clean err: {1:.3f}, tot adv err: {2:.3f}'.format(tot, tot_clean_err / tot, tot_adv_err / tot))

    elapsed = time.time() - stime
    print('elapsed:', elapsed)

    L = 50
    NUM = 101

    out_name = '{0}-{1}-{2}'.format(args.model, args.eps, args.attack)
    with open(f'auto_exp_data/{args.model}-{args.eps}-{args.attack}-stats.txt', 'w') as f:
        f.writelines([
            f'clean err: {tot_clean_err / tot}\n',
            f'adv err: {tot_adv_err / tot}\n',
            f'time: {elapsed}\n'
        ])
    with open(f'auto_exp_data/{args.model}-{args.eps}-pgd-bound.txt', 'w') as f:
        f.writelines([f'{i} {int(x)}\n' for i, x in enumerate(detail)][L: NUM])
    with open(f'auto_exp_data/{args.model}-clean.txt', 'w') as f:
        f.writelines([f'{i} {int(x)}\n' for i, x in enumerate(clean_detail)][L: NUM])

