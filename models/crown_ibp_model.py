import torch
import torch.nn as nn
from models.zoo import *
from models.test_model import get_normalize_layer

def load_model(model, path):
    model_file = path
    # model_file += "_pretrain"
    print("Loading model file", model_file)
    checkpoint = torch.load(model_file)
    if isinstance(checkpoint["state_dict"], list):
        checkpoint["state_dict"] = checkpoint["state_dict"][0]
    new_state_dict = {}
    for k in checkpoint["state_dict"].keys():
        if "prev" in k:
            pass
        else:
            new_state_dict[k] = checkpoint["state_dict"][k]
    checkpoint["state_dict"] = new_state_dict

    """
    state_dict = m.state_dict()
    state_dict.update(checkpoint["state_dict"])
    m.load_state_dict(state_dict)
    print(checkpoint["state_dict"]["__mask_layer.weight"])
    """

    model.load_state_dict(checkpoint["state_dict"])


def crown_ibp_mnist_cnn_2layer_width_2(eps):
    assert eps in ['1', '2', '3', '4']

    model = model_cnn_2layer(in_ch=1, in_dim=28, width=2, linear_size=256)
    load_model(model, f'models_weights/crown-ibp_models/mnist_0.{eps}_mnist_crown/mnist_crown/cnn_2layer_width_2_best.pth')
    model = model.cuda()
    return model


def crown_ibp_mnist_cnn_3layer_fixed_kernel_5_width_16(eps):
    assert eps in ['1', '2', '3', '4']

    model = model_cnn_3layer_fixed(in_ch=1, in_dim=28, kernel_size=5, width=16, linear_size=512)
    load_model(model, f'models_weights/crown-ibp_models/mnist_0.{eps}_mnist_crown_large/mnist_crown_large/cnn_3layer_fixed_kernel_5_width_16_best.pth')
    model = model.cuda()
    return model


def crown_ibp_cifar_cnn_2layer_width_2(eps):
    assert eps in ['2', '8']
    eps = {'2': '0.00784', '8': '0.03137'}[eps]

    model = model_cnn_2layer(in_ch=3, in_dim=32, width=2, linear_size=256)
    load_model(model, f'models_weights/crown-ibp_models/cifar_crown_{eps}/cifar_8_small/cnn_2layer_width_2_best.pth')
    model = nn.Sequential(get_normalize_layer('cifar10'), model)
    model = model.cuda()

    return model


def crown_ibp_cifar_cnn_3layer_fixed_kernel_3_width_16(eps):
    assert eps in ['2', '8']
    eps = {'2': '0.00784', '8': '0.03137'}[eps]

    model = model_cnn_3layer_fixed(in_ch=3, in_dim=32, kernel_size=3, width=16, linear_size=512)
    load_model(model, f'models_weights/crown-ibp_models/cifar_crown_large_{eps}/cifar_8_large/cnn_3layer_fixed_kernel_3_width_16_best.pth')
    model = nn.Sequential(get_normalize_layer('cifar10'), model)
    model = model.cuda()

    return model


def ibp_mnist_large(eps):
    assert eps in ['2', '4']

    model = IBP_large(in_ch=1, in_dim=28, linear_size=512)
    load_model(model, f'models_weights/models_crown-ibp_dm-large/mnist_dm-large_0.{eps}/IBP_large_best.pth')
    model = model.cuda()

    return model


def ibp_cifar_large(eps):
    assert eps in ['2', '8']

    model = IBP_large(in_ch=3, in_dim=32, linear_size=512)
    load_model(model, f'models_weights/models_crown-ibp_dm-large/cifar_dm-large_{eps}_255/IBP_large_best.pth')
    model = nn.Sequential(get_normalize_layer('cifar10'), model)
    model = model.cuda()

    return model



