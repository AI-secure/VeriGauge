#!/usr/bin/env python3
## parse_lipschitz.py
## 
## Generate figure for Lipschitz constant experiment
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import os
import sys
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

def parse_result_line(result_line, is_array = False):
    if not result_line.startswith("[L"):
        return [], []
    result_line = result_line[4:]
    elements = result_line.split(',')
    keys = []
    vals = []
    for e in elements:
        key = e.split('=')[0].strip()
        val = e.split('=')[1].strip()
        if is_array:
            array_key = key[key.find("[")+1:]
            array_key = array_key.strip()
            array_key = array_key[:-1]
            array_key = float(array_key)
            val = float(val)
            keys.append(array_key)
            vals.append(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass
            keys.append(key)
            vals.append(val)
    return keys, vals
            

def parse_single(lines):
    for i, l in enumerate(lines):
        if l.startswith("[L1] lipschitz[0.00000]"):
            # next line contains global Lipschitz constant
            result_line = l
            break
    else:
        raise(RuntimeError("result line cannot be found"))
    radius, lipschitz = parse_result_line(result_line, is_array = True)
    keys, vals = parse_result_line(lines[-1], is_array = False)
    global_lipschitz = None
    for i, k in enumerate(keys):
        if k == "global_lipschitz":
            global_lipschitz = vals[i]
    return radius, lipschitz, global_lipschitz

def get_fig_info(filename):
    filename = filename[:filename.rfind('.')]
    elements = filename.split('_')
    dataset, nlayer, activation, nhidden, alg, direction, shift = elements
    nlayer = int(nlayer[:nlayer.find('l')])
    dataset = dataset.upper() 
    if alg == "fast":
        alg = "FastLip"
    elif alg == "general":
        if direction == "-1":
            alg = "RecurJac-B"
        else:
            if shift == "1":
                alg = "RecurJac-F1"
            else:
                alg = "RecurJac-F0"
    if activation == "leaky":
        activation = "LeakyReLU"
    if activation == "relu":
        activation = "ReLU"
    return dataset, nlayer, alg, activation

def gen_legend(dataset, nlayer, alg):
    return alg

def gen_title(dataset, nlayer, activation):
    return "{}, {} layer, {}".format(dataset, nlayer, activation)

def gen_filename(dataset, nlayer, activation):
    return "{}_{}layer_{}".format(dataset, nlayer, activation)

def get_linestyle(legend):
    if "FastLip" in legend:
        return "-."
    else:
        return "-"

def get_color(legend):
    if legend == "FastLip":
        return "C1"
    if legend == "RecurJac-F0":
        return "C3"
    if legend == "RecurJac-F1":
        return "C2"
    if legend == "RecurJac-B":
        return "C0"
    return "red"

result_x = []
result_y = []
result_global = []
legends = []

for filename in sys.argv[1:]:
    print('parsing', filename)
    with open(filename) as f:
        radius, lipschitz, global_lipschitz = parse_single(f.readlines())
        result_x.append(radius)
        result_y.append(lipschitz)
        # might not finish running yet
        if global_lipschitz:
            result_global.append(global_lipschitz)
        dataset, nlayer, alg, activation = get_fig_info(filename)
        legends.append(gen_legend(dataset, nlayer, alg))
        title = gen_title(dataset, nlayer, activation)

result_global_set = set(result_global)
if len(result_global_set) > 1:
    raise(RuntimeError("Incompatible files, global lipschitz = " + str(result_line)))
horizontal_line = list(result_global_set)[0]
print("global Lipschitz constant is", horizontal_line)
our_global = max(result_y[0])

for radius, lipschitz, label in zip(result_x, result_y, legends):
    plt.loglog(radius[1:], lipschitz[1:], label = label, color = get_color(label), linestyle = get_linestyle(label))
plt.axhline(y = horizontal_line, color = 'C4', linestyle=':', label = "Global (naive)")
plt.axhline(y = our_global, color = 'C5', linestyle='--', label = "Global (ours)")
plt.title(title)
plt.xlabel('Radius of $\ell_\infty$ ball')
plt.ylabel('Lipschitz Constant Upper Bound')

plt.legend()
plt.tight_layout(pad=0.1)
plt.show()
filename = gen_filename(dataset, nlayer, activation) + ".pdf"
print("saving to", filename)
plt.savefig(filename)

