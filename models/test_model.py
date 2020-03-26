
import torch
from torch import nn

import math

from datasets import NormalizeLayer
from models.zoo import *

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.225, 0.225, 0.225]

_CIFAR10_MEAN = [0.485, 0.456, 0.406]
_CIFAR10_STDDEV = [0.225, 0.225, 0.225]

_MNIST_MEAN = [0.0]
_MNIST_STDDEV = [1.0]

# below are for randomized smoothing
# _IMAGENET_MEAN = [0.485, 0.456, 0.406]
# _IMAGENET_STDDEV = [0.229, 0.224, 0.225]
#
# _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
# _CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]
#
# _MNIST_MEAN = [0.0]
# _MNIST_STDDEV = [1.0]


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    else:
        raise Exception("Unknown dataset")


def test_cifar10(weight_path='models_weights/cifar-small-eps-8.pth'):

    model = cifar_model().cuda()
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'][0])

    normalize_layer = get_normalize_layer('cifar10')
    model = torch.nn.Sequential(normalize_layer, model)
    return model


def test_mnist(weight_path='models_weights/mnist-small-eps-3.pth'):

    model = mnist_model().cuda()
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'][0])

    normalize_layer = get_normalize_layer('mnist')
    model = torch.nn.Sequential(normalize_layer, model)
    return model


def test_cifar10_tiny(weight_path='models_weights/cifar-tiny-eps-8.pth'):

    model = cifar_model_tiny().cuda()
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'][0])
    # model.load_state_dict(checkpoint)

    normalize_layer = get_normalize_layer('cifar10')
    model = torch.nn.Sequential(normalize_layer, model)
    return model


def test_mnist_tiny(weight_path='models_weights/mnist-tiny-eps-3.pth'):

    model = mnist_model_tiny().cuda()
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'][0])
    # model.load_state_dict(checkpoint)

    normalize_layer = get_normalize_layer('mnist')
    model = torch.nn.Sequential(normalize_layer, model)
    return model
