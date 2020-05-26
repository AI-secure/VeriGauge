import torch
from torch import nn
import functools
from adaptor.basic_adaptor import NormalizeLayer
from models.test_model import Flatten
import numpy as np
from numba import jit, njit


class FlattenConv2D(nn.Module):
    """
        Transforms the 2D convolutional layer to fully-connected layer with fixed weights.
    """

    def __init__(self, orgnl_layer, shape, load_weight=True):
        super(FlattenConv2D, self).__init__()

        self.orgnl_layer = orgnl_layer
        assert isinstance(self.orgnl_layer, nn.modules.conv.Conv2d)
        # currently only support these parameters
        # moreover, suppose there is channel-wise bias
        # be future work to support more
        assert self.orgnl_layer.dilation == (1, 1)
        assert self.orgnl_layer.groups == 1

        self.orgnl_layer = orgnl_layer

        self.in_shape = shape
        self.out_shape = [self.orgnl_layer.out_channels,
                          (self.in_shape[1] + 2 * self.orgnl_layer.padding[0] - self.orgnl_layer.kernel_size[0]) // self.orgnl_layer.stride[0] + 1,
                          (self.in_shape[2] + 2 * self.orgnl_layer.padding[1] - self.orgnl_layer.kernel_size[1]) // self.orgnl_layer.stride[1] + 1]

        conv_weight = self.orgnl_layer.weight
        conv_bias = self.orgnl_layer.bias
        padding = self.orgnl_layer.padding
        stride = self.orgnl_layer.stride
        kernel_size = self.orgnl_layer.kernel_size
        weight_shape = list(conv_weight.size())

        in_cells = self.in_shape[0] * self.in_shape[1] * self.in_shape[2]
        out_cells = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        w_cells = torch.numel(conv_weight)
        b_cells = torch.numel(conv_bias)

        self.in_numel = in_cells
        self.in_features = in_cells
        self.out_numel = out_cells
        self.out_features = out_cells

        conv_weight = conv_weight.contiguous().view(w_cells)

        weight = torch.zeros((out_cells, in_cells), dtype=torch.double).cuda()
        bias = torch.zeros((out_cells,), dtype=torch.double).cuda()
        if load_weight:
            conv_weight = conv_weight.detach().cpu().numpy()
            conv_bias = conv_bias.detach().cpu().numpy()
            weight = np.zeros((out_cells, in_cells), np.double)
            bias = np.zeros((out_cells,), np.double)

            FlattenConv2D.load_weight(self.in_shape, self.out_shape, padding, stride, kernel_size, weight_shape, conv_weight, conv_bias, weight, bias)

            weight = torch.Tensor(weight).cuda()
            bias = torch.Tensor(bias).cuda()
            # cnt = 0
            # for o in range(self.out_shape[0]):
            #     for j in range(self.out_shape[1]):
            #         for k in range(self.out_shape[2]):
            #             out_idx = o * self.out_shape[1] * self.out_shape[2] + j * self.out_shape[2] + k
            #             for jj in range(kernel_size[0]):
            #                 for kk in range(kernel_size[1]):
            #                     in_x = - padding[0] + stride[0] * j + jj
            #                     in_y = - padding[1] + stride[1] * k + kk
            #                     if in_x < 0 or in_x >= self.in_shape[1] or in_y < 0 or in_y >= self.in_shape[2]:
            #                         continue
            #                     for i in range(self.in_shape[0]):
            #                         in_idx = i * self.in_shape[1] * self.in_shape[2] + in_x * self.in_shape[2] + in_y
            #                         w_idx = o * weight_shape[1] * weight_shape[2] * weight_shape[3] + \
            #                                 i * weight_shape[2] * weight_shape[3] + \
            #                                 jj * weight_shape[3] + \
            #                                 kk
            #                         weight[out_idx][in_idx] += conv_weight[w_idx]
            #                         cnt += 1
            # for o in range(self.out_shape[0]):
            #     for j in range(self.out_shape[1]):
            #         for k in range(self.out_shape[2]):
            #             out_idx = o * self.out_shape[1] * self.out_shape[2] + j * self.out_shape[2] + k
            #             bias[out_idx] = conv_bias[o]
        self.weight = weight
        self.bias = bias

    @jit(nopython=True)
    def load_weight(in_shape,
                    out_shape,
                    padding, stride, kernel_size, weight_shape,
                    conv_weight, conv_bias,
                    weight, bias):
        cnt = 0
        for o in range(out_shape[0]):
            for j in range(out_shape[1]):
                for k in range(out_shape[2]):
                    out_idx = o * out_shape[1] * out_shape[2] + j * out_shape[2] + k
                    for jj in range(kernel_size[0]):
                        for kk in range(kernel_size[1]):
                            in_x = - padding[0] + stride[0] * j + jj
                            in_y = - padding[1] + stride[1] * k + kk
                            if in_x < 0 or in_x >= in_shape[1] or in_y < 0 or in_y >= in_shape[2]:
                                continue
                            for i in range(in_shape[0]):
                                in_idx = i * in_shape[1] * in_shape[2] + in_x * in_shape[2] + in_y
                                w_idx = o * weight_shape[1] * weight_shape[2] * weight_shape[3] + \
                                        i * weight_shape[2] * weight_shape[3] + \
                                        jj * weight_shape[3] + \
                                        kk
                                weight[out_idx][in_idx] += conv_weight[w_idx]
                                cnt += 1
        for o in range(out_shape[0]):
            for j in range(out_shape[1]):
                for k in range(out_shape[2]):
                    out_idx = o * out_shape[1] * out_shape[2] + j * out_shape[2] + k
                    bias[out_idx] = conv_bias[o]
        return weight, bias

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


def model_transform(model, in_shape, load_weight=True):
    new_layers = [Flatten()]
    now_shape = in_shape
    for l in model:
        if isinstance(l, nn.modules.conv.Conv2d):
            new_layer = FlattenConv2D(l, now_shape, load_weight)
            now_shape = new_layer.out_shape
            new_layers.append(new_layer)
        elif isinstance(l, nn.modules.activation.ReLU):
            new_layers.append(nn.ReLU())
        elif isinstance(l, Flatten):
            now_shape = [functools.reduce(lambda x, y: x * y, now_shape, 1)]
        elif isinstance(l, nn.modules.linear.Linear):
            try:
                assert now_shape == [l.in_features]
            except:
                print('Error: shape size does not match.')
                raise Exception()
            now_shape = [l.out_features]
            new_layers.append(l)
        elif isinstance(l, NormalizeLayer):
            # ignore normalized layer, but need to guarantee that the std per channel is consistent
            assert max(l.orig_sds) == min(l.orig_sds)
        elif isinstance(l, nn.Dropout):
            # skip dropout layers since it is ignored in eval mode
            pass
        else:
            raise Exception(f"Unsupported layer type: {type(l)}")
    new_model = nn.Sequential(*new_layers)
    return new_model

