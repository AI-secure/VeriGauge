import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import stderr
import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import threading
import multiprocessing
import time
import getpass
import ctypes
import signal
import inspect

import datasets
from datasets import get_input_shape
import model
from models.test_model import Flatten
from models.exp_model import try_load_weight
from constants import METHOD_LIST

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict

weights = {
    'mnist': [
        ('clean', 'clean_0', 0.02),
        ('adv1', 'adv_0.1', 0.1),
        ('cadv1', 'certadv_0.1', 0.1),
        ('adv3', 'adv_0.3', 0.3),
        ('cadv3', 'certadv_0.3', 0.3)
    ],
    'cifar10': [
        ('clean', 'clean_0', 0.5/255.),
        ('adv2', 'adv_0.00784313725490196', 2.0/255.),
        ('cadv2', 'certadv_0.00784313725490196', 2.0/255.),
        ('adv8', 'adv_0.03137254901960784', 8.0/255.),
        ('cadv8', 'certadv_0.03137254901960784', 8.0/255.)
    ]
}


def param_summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    # summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # summary_str += "----------------------------------------------------------------" + "\n"
    # line_new = "{:>20}  {:>25} {:>15}".format(
    #     "Layer (type)", "Output Shape", "Param #")
    # summary_str += line_new + "\n"
    # summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        # line_new = "{:>20}  {:>25} {:>15}".format(
        #     layer,
        #     str(summary[layer]["output_shape"]),
        #     "{0:,}".format(summary[layer]["nb_params"]),
        # )
        total_params += summary[layer]["nb_params"]
        print(str(layer))
        if (str(layer).find('Flatten')) == -1 and ((str(layer).find('ReLU'))):
            total_output += np.prod(summary[layer]["output_shape"]) // batch_size
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        # summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    # summary_str += "================================================================" + "\n"
    # summary_str += "Total params: {0:,}".format(total_params) + "\n"
    # summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    # summary_str += "Non-trainable params: {0:,}".format(total_params -
    #                                                     trainable_params) + "\n"
    # summary_str += "----------------------------------------------------------------" + "\n"
    # summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    # summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    # summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    # summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    # summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return total_output, (total_params.item(), trainable_params.item())

def struc_summary(m):

    ans = ""

    cur_type = None
    cur_cnt = 0

    def refresh(s, ans, cur_type, cur_cnt):
        if s == cur_type:
            cur_cnt += 1
        else:
            if cur_type is not None:
                if len(ans) > 0:
                    ans += " $\\to$ "
                ans += f"{cur_cnt if cur_cnt > 1 else ''}{' ' if cur_cnt > 1 else ''}{cur_type}{'s' if cur_cnt > 1 else ''}"
            cur_cnt = 1
            cur_type = s
        return ans, cur_type, cur_cnt

    for layer in m:
        if isinstance(layer, Flatten):
            ans, cur_type, cur_cnt = refresh('Flatten', ans, cur_type, cur_cnt)
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            ans, cur_type, cur_cnt = refresh('Conv', ans, cur_type, cur_cnt)
        elif isinstance(layer, nn.Linear):
            ans, cur_type, cur_cnt = refresh('FC', ans, cur_type, cur_cnt)
        elif isinstance(layer, nn.ReLU):
            pass
        else:
            print(layer)
            raise NotImplementedError
    ans, cur_type, cur_cnt = refresh('end', ans, cur_type, cur_cnt)
    return ans


dataset_show_names = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR-10'
}
model_show_names = {
    'A': '\\sc{FCNNa}',
    'B': '\\sc{FCNNb}',
    'G': '\\sc{FCNNc}',
    'C': '\\sc{CNNa}',
    'D': '\\sc{CNNb}',
    'E': '\\sc{CNNc}',
    'F': '\\sc{CNNd}',
}
weight_show_names = {
    'clean': '\\texttt{reg}',
    'adv1': '\\texttt{adv1}',
    'adv3': '\\texttt{adv3}',
    'cadv1': '\\texttt{cadv1}',
    'cadv3': '\\texttt{cadv3}',
    'adv2': '\\texttt{adv2}',
    'adv8': '\\texttt{adv8}',
    'cadv2': '\\texttt{cadv2}',
    'cadv8': '\\texttt{cadv8}',
}

sources = {
    'A': '$2 \\times [20]$ from \\cite{weng2018towards,zhang2019recurjac}',
    'B': '$3 \\times [100]$ enlarged from \\cite{weng2018towards,zhang2019recurjac}',
    'C': 'Conv-Small in \\cite{wong2018provable,wong2018scaling}',
    'D': 'Half sized Conv-Large in \\cite{wong2018scaling}',
    'E': 'Conv-Large in \\cite{wong2018scaling}',
    'F': 'Double sized Conv-Large in \\cite{wong2018scaling}',
    'G': '$7 \\times [1024]$ from \\cite{weng2018towards,zhang2018efficient}'
}

SAVE_PATH = 'models_weights/exp_models'

def eliminate_redundent(tab_body, *cols):
    for c in cols:
        for i in range(len(tab_body)-1, 0, -1):
            if tab_body[i][c] == tab_body[i-1][c]:
                tab_body[i][c] = ''

def nicenum(num):
    num = f'{num}'
    s_inv = num[::-1]
    s = ''
    for i,c in enumerate(s_inv):
        s += c
        if i % 3 == 2 and i < len(s_inv) - 1:
            s += ','
    s = s[::-1]
    return '$' + s + '$'


def tex_output(tab_body, tab_head, file_handle, caption="", label=""):
    print('% This file is automatically generated by experiments/model_summary.py', file=file_handle)
    print("", file=file_handle)
    col_format = '|'.join(['c' for _ in tab_head])
    tab = [tab_head] + tab_body
    tab_str = [' & '.join(['{:>15}'.format(x) for x in line]) + " \\\\" for line in tab]
    print("""
\\begin{table}[!h]
    \\centering
    \\caption{""" + caption + """}
    \\begin{tabular}{""" + col_format + "}\n"
          + '\n'.join(tab_str) + """
    \\end{tabular}
    \\label{""" + label + """}
\\end{table}
        """, file=file_handle)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_ids', type=str, default='1')
if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids


    tab_body_1 = []
    tab_body_2 = []

    for now_ds_name in ['mnist', 'cifar10']:

        for now_model in "ABGCDEF":
            i = 0

            for now_weight_showname, now_weight_filename, now_eps in weights[now_ds_name]:

                print(now_ds_name, now_model, now_weight_showname)

                m = model.load_model('exp', now_ds_name, now_model).cuda()

                d = torch.load(f'{SAVE_PATH}/{now_ds_name}_{now_model}_{now_weight_filename}_best.pth')
                # print('acc:', d['acc'], 'robacc:', d['robacc'], 'epoch:', d['epoch'], 'normalized:', d['normalized'],
                #       'dataset:', d['dataset'], file=sys.stderr)

                model_parameters = filter(lambda p: p.requires_grad, m.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                # print(params)

                neurons, numparams = param_summary(m, get_input_shape(now_ds_name))
                # print(neurons, numparams)
                struc = struc_summary(m)
                # print(struc)

                del m

                robacc_s = f"${d['robacc'] * 100.:4.2f}\%$" if d['robacc'] > 0. else "/"

                if i == 0:
                    tab_body_1.append(
                        [
                            dataset_show_names[now_ds_name],
                            model_show_names[now_model],
                            nicenum(neurons),
                            nicenum(numparams[0]),
                            struc,
                            sources[now_model]
                        ]
                    )

                tab_body_2.append(
                    [
                        dataset_show_names[now_ds_name],
                        model_show_names[now_model],
                        weight_show_names[now_weight_showname],
                        f"${d['acc']*100.:4.2f}\%$",
                        robacc_s
                    ]
                )
                i += 1
                # break

                # print(f"{dataset_show_names[now_ds_name]} & {model_show_names[now_model]} & {neurons} & {numparams[0]} & "
                #       f"{struc} & "
                #       f"{weight_show_names[now_weight_showname]} & {d['acc']*100.:4.2f}% & {robacc_s}")

    eliminate_redundent(tab_body_1, 0,1,2,3,4)
    # for line in tab_body_1:
    #     print("{:>15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(*line))

    eliminate_redundent(tab_body_2, 0,1)

    for line in tab_body_2:
        print("{:>15} {:>15} {:>15} {:>15} {:>15}".format(*line))

    with open('experiments/tables/exp-A-models-stats.tex', 'w') as f:
        tex_output(tab_body_1, ['Dataset', 'Model', '\\# Neurons', '\\# Parameters', 'Structure', 'Source'], f,
                   caption="Statistics of the models used in \\cref{sec:exp-A}.",
                   label="table:expA-models-stats")

    with open('experiments/tables/exp-A-models-originalacc.tex', 'w') as f:
        tex_output(tab_body_2, ['Dataset', 'Model', 'Weights', 'Clean Acc.', 'PGD Adv. Acc.'], f,
                   caption="""Clean accuracy and adversarial accuracy for each set of weights.
    The clean accuracy is measured on original inputs.
    The adversarial accuracy is measured on the inputs generated by PGD attacks.
    For \\texttt{reg} weights, due to low robustness, we do not report adverasarial accuracy.
    For \\texttt{adv1} and \\texttt{cadv1}, the adversarial examples are bounded by radius $\epsilon=0.1$ under $\\cL_\\infty$ ball.
    For \\texttt{adv3} and \\texttt{cadv3}, the adversarial examples are bounded by radius $\epsilon=0.3$ under $\\cL_\\infty$ ball.""",
                   label="table:expA-models-originalacc")

    print('Finish!', file=stderr)


