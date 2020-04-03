#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clever.py

Compute CLEVER score using collected Lipschitz constants

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

import os
import sys
import glob
from functools import partial
from multiprocessing import Pool
import scipy
import scipy.io as sio
from scipy.stats import weibull_min
import scipy.optimize
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# We observe that the scipy.optimize.fmin optimizer (using Nelder–Mead method)
# sometimes diverges to very large parameters a, b and c. Thus, we add a very
# small regularization to the MLE optimization process to avoid this divergence
def fmin_with_reg(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None, shape_reg = 0.01):
    # print('my optimier with shape regularizer = {}'.format(shape_reg))
    def func_with_reg(theta, x):
        shape = theta[2]
        log_likelyhood = func(theta, x)
        reg = shape_reg * shape * shape
        # penalize the shape parameter
        return log_likelyhood + reg
    return scipy.optimize.fmin(func_with_reg, x0, args, xtol, ftol, maxiter, maxfun,
         full_output, disp, retall, callback, initial_simplex)

# fit using weibull_min.fit and run a K-S test
def fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, optimizer, c_i):
    [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    loc = - loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pVal = scipy.stats.kstest(-sample, 'weibull_min', args = (c, loc, scale))
    return c, loc, scale, ks, pVal

def plot_weibull(sample,c,loc,scale,ks,pVal,p,q,figname = "Lily_weibull_test.png"):
    
    # compare the sample histogram and fitting result
    fig, ax = plt.subplots(1,1)
    
    x = np.linspace(-1.01*max(sample),-0.99*min(sample),100);
    ax.plot(x,weibull_min.pdf(x,c,loc,scale),'r-',label='fitted pdf '+p+'-bnd')
    ax.hist(-sample, normed=True, bins=20, histtype='stepfilled')
    ax.legend(loc='best', frameon=False)
    plt.xlabel('-Lips_'+q)
    plt.ylabel('pdf')
    plt.title('c = {:.2f}, loc = {:.2f}, scale = {:.2f}, ks = {:.2f}, pVal = {:.2f}'.format(c,loc,scale,ks,pVal))
    plt.savefig(figname)
    plt.close()
    #model = figname.split("_")[1]    
    #plt.savefig('./debug/'+model+'/'+figname)    
    #plt.show() # can be used to pause the program

# We observe than the MLE estimator in scipy sometimes can converge to a bad
# value if the inital shape parameter c is too far from the true value. Thus we
# test a few different initializations and choose the one with best p-value all
# the initializations are tested in parallel; remove some of them to speedup
# computation.
# c_init = [0.01,0.1,0.5,1,5,10,20,50,70,100,200]
c_init = [0.1,1,5,10,20,50,100]

def get_best_weibull_fit(sample, pool, use_reg = False, shape_reg = 0.01):
    # initialize dictionary to save the fitting results
    fitted_paras = {"c":[], "loc":[], "scale": [], "ks": [], "pVal": []}
    # reshape the data into a better range 
    # this helps the MLE solver find the solution easier
    loc_shift = np.amax(sample)
    dist_range = np.amax(sample) - np.amin(sample)
    # if dist_range > 2.5:
    shape_rescale = dist_range
    # else:
    #     shape_rescale = 1.0
    print("shape rescale = {}".format(shape_rescale))
    rescaled_sample = np.copy(sample)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale

    print("loc_shift = {}".format(loc_shift))
    ##print("rescaled_sample = {}".format(rescaled_sample))

    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution
    if use_reg:
        results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, partial(fmin_with_reg, shape_reg = shape_reg)), c_init)
    else:
        results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, scipy.optimize.fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print("[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(c_i,c,loc,scale,ks,pVal,loc_shift))
        
        ## plot every fitted result
        #plot_weibull(sample,c,loc,scale,ks,pVal,p)
        
        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)
    
    
    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    if np.isnan(max_pVal) or max_pVal < 0.001:
        print("ill-conditioned samples. Using maximum sample value.")
        # handle the ill conditioned case
        return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)
    
    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]
    
    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best
    

# G_max is the input array of max values
# Return the Weibull position parameter
def get_lipschitz_estimate(G_max, pool, norm = "L2", use_reg = False, shape_reg = 0.01):
    c_init, c, loc, scale, ks, pVal = get_best_weibull_fit(G_max, pool, use_reg, shape_reg)
    
    # the norm here is Lipschitz constant norm, not the bound's norm
    if norm == "L1":
        p = "i"; q = "1"
    elif norm == "L2":
        p = "2"; q = "2"
    elif norm == "Li":
        p = "1"; q = "i"
    else:
        print("Lipschitz norm is not in 1, 2, i!")
    
    return {'Lips_est':-loc, 'shape':c, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}
    #return np.max(G_max)

# file name contains some information, like true_id, true_label and target_label
def parse_filename(filename):
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    name_arr = name.split('_')
    Nsamp = int(name_arr[0])
    Niters = int(name_arr[1])
    true_id = int(name_arr[2])
    true_label = int(name_arr[3])
    target_label = int(name_arr[4])
    image_info = name_arr[5]
    activation = name_arr[6]
    order = name_arr[7][-1]
    return Nsamp, Niters, true_id, true_label, target_label, image_info, activation, order

if __name__ == "__main__":
    # parse command line parameters
    parser = argparse.ArgumentParser(description='Compute CLEVER scores using collected gradient norm data.', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_folder', help='data folder path')
    parser.add_argument('--min', dest='reduce_op', action='store_const',
                    default=lambda x: sum(x) / len(x) if len(x) > 0 else 0, const=min,
                    help='report min of all CLEVER scores instead of avg')
    parser.add_argument('--user_type', 
                default="",
                help='replace user type with string, used for ImageNet data processing')
    parser.add_argument('--use_slope',
                action="store_true",
                help='report slope estimate. To use this option, collect_gradients.py needs to be run with --compute_slope')
    parser.add_argument('--untargeted',
                action="store_true",
                help='process untargeted attack results (for MNIST and CIFAR)')
    parser.add_argument('--num_samples',
                type=int,
                default=0,
                help='the number of samples to use. Default 0 is to use all samples')
    parser.add_argument('--num_images',
                type=int,
                default=0,
                help='number of images to use, 0 to use all images')
    parser.add_argument('--shape_reg',
                default=0.01,
                type=float,
                help='to avoid the MLE solver in Scipy to diverge, we add a small regularization (default 0.01 is sufficient)')
    parser.add_argument('--nthreads',
                default=0,
                type=int,
                help='number of threads (default is len(c_init)+1)')
    parser.add_argument('--plot_dir',
                default='',
                help='output path for weibull fit figures (empty to disable)')
    parser.add_argument('--method', 
                default="mle_reg",
                choices=['mle','mle_reg','maxsamp'],
                help='Fitting algorithm. Please use mle_reg for best results')
    args = vars(parser.parse_args())
    reduce_op = args['reduce_op']
    if args['plot_dir']:
        os.system("mkdir -p " + args['plot_dir'])
    print(args)

    # create thread pool
    if args['nthreads'] == 0:
        args['nthreads'] = len(c_init) + 1
    print("using {} threads".format(args['nthreads']))
    pool = Pool(processes = args['nthreads'])
    # pool = Pool(1)
    # used for asynchronous plotting in background
    plot_res = None

    # get a list of all '.mat' files in folder
    file_list = glob.glob(args['data_folder'] + '/**/*.mat', recursive = True)
    # sort by image ID, then by information (least likely, random, top-2)
    file_list = sorted(file_list, key = lambda x: (parse_filename(x)[2], parse_filename(x)[5]))
    # get the first num_images files
    if args['num_images']:
        file_list = file_list[:args['num_images']]

    if args['untargeted']:
        bounds = {}
        # bounds will be inserted per image
    else:
        # aggregate information for three different types: least, random and top2
        # each has three bounds: L1, L2, and Linf
        bounds = {"least" : [[], [], []],
                  "random": [[], [], []],
                  "top2"  : [[], [], []]}

    for fname in file_list:
        nsamps, niters, true_id, true_label, target_label, img_info, activation, order = parse_filename(fname)

        # keys in mat:
        # ['Li_max', 'pred', 'G1_max', 'g_x0', 'path', 'info', 'G2_max', 'true_label', 'args', 'L1_max', 'Gi_max', 'L2_max', 'id', 'target_label']
        mat = sio.loadmat(fname)
        print('loading {}'.format(fname))
        
        if order == "1" and args['use_slope']:
            G1_max = np.squeeze(mat['L1_max'])
            G2_max = np.squeeze(mat['L2_max'])
            Gi_max = np.squeeze(mat['Li_max'])
        elif order == "1":
            G1_max = np.squeeze(mat['G1_max'])
            G2_max = np.squeeze(mat['G2_max'])
            Gi_max = np.squeeze(mat['Gi_max'])
        elif order == "2":
            """ For Jun 25 experiments: forgot to save g_x0_grad_2_norm, so rerun a 1 sample 1 iterations cases "1_1_*.mat" and load g_x0_grad_2_norm from it
            fname_ref = os.path.dirname(fname)+'_1/'+"1_1_"+str(true_id)+"_"+str(true_label)+"_"+str(target_label)+"_"+img_info+"_"+activation+"_order2.mat"
            ##fname_ref = 'lipschitz_mat/mnist_normal/'+"1_1_"+str(true_id)+"_"+str(true_label)+"_"+str(target_label)+"_"+img_info+"_"+activation+"_order2.mat"
            print("loading {}".format(fname_ref))
            mat_ref = sio.loadmat(fname_ref)
            g_x0_grad_2_norm = np.squeeze(mat_ref['g_x0_grad_2_norm'])
            print("g_x0_grad_2_norm = {}".format(g_x0_grad_2_norm))
            
            #import time
            #time.sleep(30)
            """
            G2_max = np.abs(np.squeeze(mat['H2_max'])) # forgot to add abs when save in mat file
            G1_max = -1*np.empty_like(G2_max) # currently only implemented 2nd order bound for p = 2
            Gi_max = -1*np.empty_like(G2_max)
            g_x0_grad_2_norm = np.squeeze(mat['g_x0_grad_2_norm'])
        else:
            raise RuntimeError('!!! order is {}'.format(order))

        if args['num_samples'] != 0:
            prev_len = len(G1_max)
            G1_max = G1_max[:args['num_samples']]
            G2_max = G2_max[:args['num_samples']]
            Gi_max = Gi_max[:args['num_samples']]
            print('Using {} out of {} total samples'.format(len(G1_max), prev_len))
        g_x0 = np.squeeze(mat['g_x0'])
        target_label = np.squeeze(mat['target_label'])
        true_id = np.squeeze(mat['id'])
        true_label = np.squeeze(mat['true_label'])
        img_info = mat['info'][0]
        if args['user_type'] != "" and img_info == "user":
            img_info = args['user_type']
        
        # get the filename (.mat)
        print('[Filename] {}'.format(fname))       
        # get the model name (inception, cifar_2-layer)
        possible_names = ["mnist", "cifar", "mobilenet", "inception", "resnet"]
        model = "unknown"
        for path_seg in args["data_folder"].split("/"):
            for n in possible_names:
                if n in path_seg:
                    model = path_seg.replace('_', '-')
                    break
        # model = args["data_folder"].split("/")[1] 
        
        if args['num_samples'] == 0: # default, use all G1_max
            figname = 'Fig_'+model+'_'+img_info+'_'+str(true_id)+'_'+str(true_label)+'_'+str(target_label)+'_Nsamp_'+str(len(G1_max));  
        elif args['num_samples'] <= len(G1_max) and args['num_samples'] > 0:
            figname = 'Fig_'+model+'_'+img_info+'_'+str(true_id)+'_'+str(true_label)+'_'+str(target_label)+'_Nsamp_'+str(args['num_samples']);
        else:
            print('Warning!! Input arg num_samp = {} exceed len(G1_max) in data_process.py'.format(args['num_samples']))
            continue
            
       
        if args['use_slope']:
            figname = figname + '_slope'

        if args['plot_dir']:
            figname = os.path.join(args['plot_dir'], figname)
            # figname 
            print('[Figname] {}'.format(figname))
        else:
            # disable debugging figure
            figname = ""
        
        
        
        if args['method'] == "maxsamp":
            if order == "1":
                Est_G1 = {'Lips_est': max(G1_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}
                Est_G2 = {'Lips_est': max(G2_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}
                Est_Gi = {'Lips_est': max(Gi_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}
            else: # currently only compare bounds in L2 for both order = 1 and order = 2
                Est_G2 = {'Lips_est': max(G2_max), 'shape': -1, 'loc': -1, 'scale': -1, 'ks': -1, 'pVal': -1}
                Est_G1 = Est_G2
                Est_Gi = Est_G2

        elif args['method'] == "mle":
            # estimate Lipschitz constant: Est_G1 is a dictionary containing Lips_est and weibull paras        
            if order == "1": 
                Est_G1 = get_lipschitz_estimate(G1_max, "L1", figname)
                Est_G2 = get_lipschitz_estimate(G2_max, "L2", figname)
                Est_Gi = get_lipschitz_estimate(Gi_max, "Li", figname)
            else: # currently only compare bounds in L2 for both order = 1 and order = 2
                Est_G2 = get_lipschitz_estimate(G2_max, "L2", figname)
                Est_G1 = Est_G2 # haven't implemented
                Est_Gi = Est_G2 # haven't implemented

        elif args['method'] == "mle_reg":
            
            if order == "1":
                print('estimating L1...')
                Est_G1 = get_lipschitz_estimate(G1_max, "L1", figname, True, args['shape_reg'])
                print('estimating L2...')
                Est_G2 = get_lipschitz_estimate(G2_max, "L2", figname, True, args['shape_reg'])
                print('estimating Li...')
                Est_Gi = get_lipschitz_estimate(Gi_max, "Li", figname, True, args['shape_reg'])
            else: # currently only compare bounds in L2 for both order = 1 and order = 2
                print('estimating L2...')
                Est_G2 = get_lipschitz_estimate(G2_max, "L2", figname, True, args['shape_reg'])
                Est_G1 = Est_G2
                Est_Gi = Est_G1
        else:
            raise RuntimeError("method not supported")
                
        # the estimated Lipschitz constant
        Lip_G1 = Est_G1['Lips_est']
        Lip_G2 = Est_G2['Lips_est']
        Lip_Gi = Est_Gi['Lips_est']
        
        # the estimated shape parameter (c) in Weibull distn 
        shape_G1 = Est_G1['shape']
        shape_G2 = Est_G2['shape']
        shape_Gi = Est_Gi['shape']
        
        # the estimated loc parameters in Weibull distn
        loc_G1 = Est_G1['loc']
        loc_G2 = Est_G2['loc']
        loc_Gi = Est_Gi['loc']
              
        # the estimated scale parameters in Weibull distn
        scale_G1 = Est_G1['scale']
        scale_G2 = Est_G2['scale']
        scale_Gi = Est_Gi['scale']
        
        # the computed ks score
        ks_G1 = Est_G1['ks']
        ks_G2 = Est_G2['ks']
        ks_Gi = Est_Gi['ks']
        
        # the computed pVal
        pVal_G1 = Est_G1['pVal']
        pVal_G2 = Est_G2['pVal']
        pVal_Gi = Est_Gi['pVal']
        
        
        # compute robustness bound
        if order == "1": 
            bnd_L1 = g_x0 / Lip_Gi
            bnd_L2 = g_x0 / Lip_G2
            bnd_Li = g_x0 / Lip_G1
        else:
            bnd_L2 = (-g_x0_grad_2_norm + np.sqrt(g_x0_grad_2_norm**2+2*g_x0*Lip_G2))/Lip_G2
            bnd_L1 = bnd_L2 # haven't implemented 
            bnd_Li = bnd_L2 # haven't implemented
        # save bound of each image
        if args['untargeted']:
            true_id = int(true_id)
            if true_id not in bounds:
                bounds[true_id] = [[], [], []]
            bounds[true_id][0].append(bnd_L1)
            bounds[true_id][1].append(bnd_L2)
            bounds[true_id][2].append(bnd_Li)
        else:
            bounds[img_info][0].append(bnd_L1)
            bounds[img_info][1].append(bnd_L2)
            bounds[img_info][2].append(bnd_Li)
        
        # original data_process mode
        #print('[STATS][L1] id = {}, true_label = {}, target_label = {}, info = {}, bnd_L1 = {:.5g}, bnd_L2 = {:.5g}, bnd_Li = {:.5g}'.format(true_id, true_label, target_label, img_info, bnd_L1, bnd_L2, bnd_Li))
        
        bndnorm_L1 = "1";
        bndnorm_L2 = "2";
        bndnorm_Li = "i";
        
        # if use g_x0 = {:.5g}.format(g_x0), then it will have type error. Not sure why yet. 
        #print('g_x0 = '+str(g_x0))
        
        
        if args['method'] == "maxsamp":
            if order == "1":
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_L1, bnd_L1))
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_L2, bnd_L2))
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_Li, bnd_Li))
            else: # currently only compare L2 bound
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_L2, bnd_L2))
        
        elif args['method'] == "mle" or args['method'] == "mle_reg":
            if order == "1":
                # estimate Lipschitz constant: Est_G1 is a dictionary containing Lips_est and weibull paras        
                # current debug mode: bound_L1 corresponds to Gi, bound_L2 corresponds to G2, bound_Li corresponds to G1
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, scale = {:.5g}, g_x0 = {}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_L1, bnd_L1, ks_Gi, pVal_Gi, shape_Gi, loc_Gi, scale_Gi, g_x0))
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, scale = {:.5g}, g_x0 = {}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_L2, bnd_L2, ks_G2, pVal_G2, shape_G2, loc_G2, scale_G2, g_x0))
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, scale = {:.5g}, g_x0 = {}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_Li, bnd_Li, ks_G1, pVal_G1, shape_G1, loc_G1, scale_G1, g_x0))      
            else: # currently only compare L2 bound
                print('[DEBUG][L1] id = {}, true_label = {}, target_label = {}, info = {}, nsamps = {}, niters = {}, bnd_norm = {}, bnd = {:.5g}, ks = {:.5g}, pVal = {:.5g}, shape = {:.5g}, loc = {:.5g}, scale = {:.5g}, g_x0 = {}'.format(true_id, true_label, target_label, img_info, nsamps, niters, bndnorm_L2, bnd_L2, ks_G2, pVal_G2, shape_G2, loc_G2, scale_G2, g_x0))

        else:
            raise RuntimeError("method not supported")
        
        sys.stdout.flush()
        
    if args['untargeted']:
        clever_L1s = []
        clever_L2s = []
        clever_Lis = []
        for true_id, true_id_bounds in bounds.items():
            img_clever_L1 = min(true_id_bounds[0])
            img_clever_L2 = min(true_id_bounds[1])
            img_clever_Li = min(true_id_bounds[2])
            n_classes = len(true_id_bounds[0]) + 1
            assert len(true_id_bounds[0]) == len(true_id_bounds[2])
            assert len(true_id_bounds[1]) == len(true_id_bounds[2])
            print('[STATS][L1] image = {:3d}, n_classes = {:3d}, clever_L1 = {:.5g}, clever_L2 = {:.5g}, clever_Li = {:.5g}'.format(true_id, n_classes, img_clever_L1, img_clever_L2, img_clever_Li))
            clever_L1s.append(img_clever_L1)
            clever_L2s.append(img_clever_L2)
            clever_Lis.append(img_clever_Li)
        info = "untargeted"
        clever_L1 = reduce_op(clever_L1s)
        clever_L2 = reduce_op(clever_L2s)
        clever_Li = reduce_op(clever_Lis)
        print('[STATS][L0] info = {}, {}_clever_L1 = {:.5g}, {}_clever_L2 = {:.5g}, {}_clever_Li = {:.5g}'.format(info, info, clever_L1, info, clever_L2, info, clever_Li))
    else:
        # print min/average bound
        for info, info_bounds in bounds.items():
            # reduce each array to a single number (min or avg)
            clever_L1 = reduce_op(info_bounds[0])
            clever_L2 = reduce_op(info_bounds[1])
            clever_Li = reduce_op(info_bounds[2])
            if order == "1":
                print('[STATS][L0] info = {}, {}_clever_L1 = {:.5g}, {}_clever_L2 = {:.5g}, {}_clever_Li = {:.5g}'.format(info, info, clever_L1, info, clever_L2, info, clever_Li))
            else: # currently only compare L2 bound for both order = 1 and order = 2
                print('[STATS][L0] info = {}, {}_clever_L2 = {:.5g}'.format(info, info, clever_L2))

            sys.stdout.flush()


    # shutdown thread pool
    pool.close()
    pool.join()
