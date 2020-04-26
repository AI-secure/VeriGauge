import sys

import torch.nn as nn
import torch
import math

from models.zoo import Flatten


def get_input_shape(dataset: str):
    """Return a list of integer indicating the input shape as (num_channel, height, weight)"""
    if dataset == "imagenet":
        return (3, 224, 224)
    elif dataset == 'cifar10':
        return (3, 32, 32)
    elif dataset == 'mnist':
        return (1, 28, 28)

def in_cells(dataset):
    ans = 1
    for i in get_input_shape(dataset):
        ans *= i
    return ans

# ========= load from saved weights =========

SAVE_PATH = 'models_weights/exp_models'

def try_load_weight(m, name):
    try:
        d = torch.load(f'{SAVE_PATH}/{name}.pth')
        print('acc:', d['acc'], 'robacc:', d['robacc'], 'epoch:', d['epoch'], 'normalized:', d['normalized'], 'dataset:', d['dataset'], file=sys.stderr)
        m.load_state_dict(d['state_dict'])
        from models.test_model import get_normalize_layer
        if d['normalized']:
            # note: the original save model does not contain normalized layer
            m = nn.Sequential(get_normalize_layer(d['dataset']), m)
        print(f'load from {name}', file=sys.stderr)
        return m, True
    except:
        print('no load', file=sys.stderr)
        return m, False

# def eliminate_batchnorm(m):
#     """
#         Eliminate the batch norm 1d which is not supported by some
#     :param m:
#     :return:
#     """
#     pre_layer = None
#     new_m = list()
#     for l in m:
#         if isinstance(l, nn.BatchNorm1d):
#             assert isinstance(l, )
#             print(l.weight)
#             print(l.bias)
#             l.

# ========= model FC small (A) ==========
def two_layer_fc20(ds):
    assert ds in ['mnist', 'cifar10'], 'unknown dataset name'
    model = nn.Sequential(
        Flatten(),
        nn.Linear(in_cells(ds), 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )
    return model

# ========= model FC medium (B) ==========
def three_layer_fc100(ds):
    assert ds in ['mnist', 'cifar10'], 'unknown dataset name'
    model = nn.Sequential(
        Flatten(),
        nn.Linear(in_cells(ds), 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


# ========= model Conv Small (C) ==========

def mnist_conv_small():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv_small():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

# ========= model Conv Medium (D) ==========

def mnist_conv_medium():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,512),
        nn.ReLU(),
        # nn.Linear(512,512),
        # nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #         m.bias.data.zero_()
    # return model


def cifar_conv_medium():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,512),
        nn.ReLU(),
        # nn.Linear(512,512),
        # nn.ReLU(),
        nn.Linear(512,10)
    )
    # return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


# ========= model Conv Large (E) ==========

def mnist_conv_large():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


def cifar_conv_large():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


# ========= model conv super (F) ==========

def conv_super(ds, linear_size=512):
    assert ds in ['mnist', 'cifar10'], 'unknown dataset name'
    in_ch, h, w = get_input_shape(ds)
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((h//2) * (w//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


# ========= model FC Super (G) ==========
def seven_layer_fc1024(ds):
    assert ds in ['mnist', 'cifar10'], 'unknown dataset name'
    model = nn.Sequential(
        Flatten(),
        nn.Linear(in_cells(ds), 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    return model


