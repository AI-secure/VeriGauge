#!/usr/bin/env python3
## parse_landscape.py
## 
## Generate figure for optimization landscape experiment
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
        if l.startswith("[L0] model"):
            # next line contains global Lipschitz constant
            result_line = l
            break
    else:
        raise(RuntimeError("result line cannot be found"))
    keys, vals = parse_result_line(result_line, is_array = False)
    for i, k in enumerate(keys):
        if k == "avg_max_eps":
            avg_max_eps = vals[i]
    return avg_max_eps

def get_fig_info(filename):
    filename = filename[:filename.rfind('.')]
    elements = filename.split('_')
    dataset, nlayer, activation, nhidden, norm = elements
    nlayer = int(nlayer[:nlayer.find('l')])
    return dataset, nlayer, activation, norm

def gen_title(dataset, nlayer, activation):
    return "{}, {} layer, {}".format(dataset, nlayer, activation)

result_x_2 = []
result_y_2 = []
result_x_i = []
result_y_i = []

for filename in sys.argv[1:]:
    print('parsing', filename)
    with open(filename) as f:
        avg_max_eps = parse_single(f.readlines())
        dataset, nlayer, activation, norm = get_fig_info(filename)
        if norm == "2":
            result_x_2.append(nlayer)
            result_y_2.append(avg_max_eps)
        elif norm == "i":
            result_x_i.append(nlayer)
            result_y_i.append(avg_max_eps)
        # check if all logs have the same norm
        print("norm = {}, nlayer = {}, avg_max_eps = {}".format(norm, nlayer, avg_max_eps))

# sort by layer
result_x_2, result_y_2 = [list(x) for x in zip(*sorted(zip(result_x_2, result_y_2), key=lambda pair: pair[0]))]
result_x_i, result_y_i = [list(x) for x in zip(*sorted(zip(result_x_i, result_y_i), key=lambda pair: pair[0]))]

print(result_x_2)
print(result_y_2)
print(result_x_i)
print(result_y_i)
fig, ax1 = plt.subplots()

ax1.plot(result_x_2, result_y_2, color = 'b', linestyle = "-", label = "$\ell_2$")
ax1.set_xlabel('Network depth')
ax1.set_ylabel('Radius of $\ell_2$ ball, $R^*_2$')
ax1.set_ylim(0, 1.5)
ax1.tick_params('y', colors='b')
ax1.legend(loc=1, frameon=False)

ax2 = ax1.twinx()
ax2.plot(result_x_i, result_y_i, color = 'r', linestyle = "--", label = "$\ell_\infty$")
ax2.set_ylabel('Radius of $\ell_\infty$ ball, $R^*_\infty$')
ax2.set_ylim(0, 0.1)
ax2.tick_params('y', colors='r')
ax2.legend(loc=1, bbox_to_anchor=(0, 0.9, 1, 0), frameon=False)
plt.title("MNIST, 2-10 layers, LeakyReLU", fontsize=18)

plt.tight_layout(pad=0.1)
plt.show()
plt.savefig('landscape.pdf')

