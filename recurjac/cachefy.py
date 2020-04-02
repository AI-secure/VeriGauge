#!/usr/bin/env python3
## cachefy.py
## 
## Add or remove numba "cache" decorators
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import sys
import glob

filelist = glob.glob('*.py')

def add_cache(l):
    if l.find("cache=True") >= 0: # already patched
        return l
    if l[-1] == "t": #njit or jit
        l += "(cache=True)"
    elif l[-1] == ")": # some other parameters exist
        if "cache" not in l:
            l = l[:-1] + ",cache=True)"
    return l

def remove_cache(l):
    if l[-1] == ")": # some parameters exist
        l = l.replace("cache=True", "")
        # remove the extra ,
        s = l.strip()[:-1].strip()
        if s[-1] == ",":
            l = s[:-1] + ")"
    # remove empty ()
    if l[-2:] == "()":
        l = l[:-2]
    return l

func = add_cache
if len(sys.argv) > 1:
    if sys.argv[1] == "-u":
        func = remove_cache

for pyfile in filelist:
    with open(pyfile) as f:
        lines = f.readlines()
        print("processing", pyfile)
        write = False
        for i in range(len(lines)):
            l = lines[i]
            l = l.strip()
            if l.startswith("@jit") or l.startswith("@njit"):
                new_l = func(l)
                if new_l != l:
                    print('line {:5d}: "{}" -> "{}"'.format(i+1, l, new_l))
                    lines[i] = new_l + "\n"
                    write = True
                else:
                    print('line {:5d}: "{}" (patched)'.format(i+1, l))
    if write:
        print("updating", pyfile)
        with open(pyfile, "w") as f:
            f.writelines(lines)


