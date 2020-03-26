import math
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

# ======= test models =======

def mnist_model():
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

def cifar_model():
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


def cifar_model_tiny():
    model = nn.Sequential(
        nn.Conv2d(3, 3, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_model_tiny():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# ======= IBP models =======


def IBP_large(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


def IBP_debug(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//4) * (in_dim//4) * 1, 10),
    )
    return model

# ======= CROWN-IBP models =======


# MLP model, each layer has the same number of neuron
# parameter in_dim: input image dimension, 784 for MNIST and 1024 for CIFAR
# parameter layer: number of layers
# parameter neuron: number of neurons per layer
def model_mlp_uniform(in_dim, layer, neurons, out_dim = 10):
    assert layer >= 2
    neurons = [neurons] * (layer - 1)
    return model_mlp_any(in_dim, neurons, out_dim)

# MLP model, each layer has the different number of neurons
# parameter in_dim: input image dimension, 784 for MNIST and 1024 for CIFAR
# parameter neurons: a list of neurons for each layer
def model_mlp_any(in_dim, neurons, out_dim = 10):
    assert len(neurons) >= 1
    # input layer
    units = [Flatten(), nn.Linear(in_dim, neurons[0])]
    prev = neurons[0]
    # intermediate layers
    for n in neurons[1:]:
        units.append(nn.ReLU())
        units.append(nn.Linear(prev, n))
        prev = n
    # output layer
    units.append(nn.ReLU())
    units.append(nn.Linear(neurons[-1], out_dim))
    #print(units)
    return nn.Sequential(*units)

def model_cnn_1layer(in_ch, in_dim, width):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 8*width, 4, stride=4),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),10),
    )
    return model


# CNN, small 2-layer (kernel size fixed to 4) TODO: use other kernel size
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter width: width multiplier
def model_cnn_2layer(in_ch, in_dim, width, linear_size=128):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

# CNN, relatively small 3-layer
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter kernel_size: convolution kernel size, 3 or 5
# parameter width: width multiplier
def model_cnn_3layer_fixed(in_ch, in_dim, kernel_size, width, linear_size = None):
    if linear_size is None:
        linear_size = width * 64
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*h*h, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

# CNN, relatively large 4-layer
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter width: width multiplier
# TODO: the model we used before is equvalent to width=8, TOO LARGE!
# TODO: use different kernel size in this model
def model_cnn_4layer(in_ch, in_dim, width, linear_size):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def model_cnn_10layer(in_ch, in_dim, width):
    model = nn.Sequential(
        # input 32*32*3
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 32*32*4
        nn.Conv2d(4*width, 8*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 16*16*8
        nn.Conv2d(8*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 16*16*8
        nn.Conv2d(8*width, 16*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 8*8*16
        nn.Conv2d(16*width, 16*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 8*8*16
        nn.Conv2d(16*width, 32*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 4*4*32
        nn.Conv2d(32*width, 32*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 4*4*32
        nn.Conv2d(32*width, 64*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 2*2*64
        Flatten(),
        nn.Linear(2*2*64*width,10)
    )
    return model
