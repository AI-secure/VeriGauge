#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_gradients.py

Front end for collecting maximum gradient norm samples

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

from __future__ import division

import glob
import numpy as np
import scipy.io as sio
import random
import time
import sys
import os
from functools import partial
from multiprocessing import Pool
import scipy
from scipy.stats import weibull_min
import scipy.optimize

from CLEVER.estimate_gradient_norm import EstimateLipschitz
from CLEVER.clever import get_lipschitz_estimate, parse_filename
from utils import generate_data

def collect_gradients(dataset, model_name, norm, numimg=10):
    random.seed(1215)
    np.random.seed(1215)

    # create output directory
    os.system("mkdir -p {}/{}_{}".format('lipschitz_mat', dataset, model_name))
    
    # create a Lipschitz estimator class (initial it early to save multiprocessing memory)
    clever_estimator = EstimateLipschitz(sess=None, nthreads=0)

    ids = None
    target_classes = None
    target_type = 0b0010

    Nsamp = 128
    Niters = 100
    
    import tensorflow as tf
    from setup_cifar import CIFAR
    from setup_mnist import MNIST
    from setup_tinyimagenet import tinyImagenet

    np.random.seed(1215)
    tf.set_random_seed(1215)
    random.seed(1215)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        clever_estimator.sess = sess
        # returns the input tensor and output prediction vector
        img, output, model = clever_estimator.load_model(dataset, model_name, batch_size = 0, compute_slope = False, order = 1)
        # load dataset
        datasets_loader = {"mnist": MNIST, "cifar": CIFAR, "tinyimagenet": tinyImagenet}
        data = datasets_loader[dataset]()
        # for prediction
        predictor = lambda x: np.squeeze(sess.run(output, feed_dict = {img: x}))
        # generate target images
        inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=numimg, targeted=True, 
                                    random_and_least_likely = True, target_type = target_type, predictor=model.model.predict, start=0)

        timestart = time.time()
        print("got {} images".format(inputs.shape))
        total = 0
        for i, input_img in enumerate(inputs):
            # original_predict = np.squeeze(sess.run(output, feed_dict = {img: [input_img]}))
            print("processing image {}".format(i))
            original_predict = predictor([input_img])
            true_label = np.argmax(true_labels[i])
            predicted_label = np.argmax(original_predict)
            least_likely_label = np.argmin(original_predict)
            original_prob = np.sort(original_predict)
            original_class = np.argsort(original_predict)
            print("Top-10 classifications:", original_class[-1:-11:-1])
            print("True label:", true_label)
            print("Top-10 probabilities/logits:", original_prob[-1:-11:-1])
            print("Most unlikely classifications:", original_class[:10])
            print("Most unlikely probabilities/logits:", original_prob[:10])
            if true_label != predicted_label:
                print("[WARNING] This image is classfied wrongly by the classifier! Skipping!")
                continue
            total += 1
            # set target class
            target_label = np.argmax(targets[i]);
            print('Target class: ', target_label)
            sys.stdout.flush()
           
            [L2_max,L1_max,Li_max,G2_max,G1_max,Gi_max,g_x0,pred] = clever_estimator.estimate(input_img, true_label, target_label, Nsamp, Niters, 'l2', '', 1)
            print("[STATS][L1] total = {}, seq = {}, id = {}, time = {:.3f}, true_class = {}, target_class = {}, info = {}".format(total, i, true_ids[i], time.time() - timestart, true_label, target_label, img_info[i]))

            
            # save to sampling results to matlab ;)
            mat_path = "{}/{}_{}/{}_{}_{}_{}_{}_{}_{}_order{}".format('./lipschitz_mat', dataset, model_name, Nsamp, Niters, true_ids[i], true_label, target_label, img_info[i], 'relu', 1)
            save_dict = {'L2_max': L2_max, 'L1_max': L1_max, 'Li_max': Li_max, 'G2_max': G2_max, 'G1_max': G1_max, 'Gi_max': Gi_max, 'pred': pred, 'g_x0': g_x0, 'id': true_ids[i], 'true_label': true_label, 'target_label': target_label, 'info':img_info[i], 'path': mat_path}
            sio.savemat(mat_path, save_dict)
            print('saved to', mat_path)
            sys.stdout.flush()

    c_init = [0.1,1,5,10,20,50,100]

    # create thread pool
    nthreads = len(c_init) + 1
    print("using {} threads".format(nthreads))
    pool = Pool(processes = nthreads)
    # pool = Pool(1)
    # used for asynchronous plotting in background
    plot_res = None

    # get a list of all '.mat' files in folder
    file_list = glob.glob('lipschitz_mat/'+dataset+'_'+model_name+'/**/*.mat', recursive = True)
    # sort by image ID, then by information (least likely, random, top-2)
    file_list = sorted(file_list, key = lambda x: (parse_filename(x)[2], parse_filename(x)[5]))


    # aggregate information for three different types: least, random and top2
    # each has three bounds: L1, L2, and Linf
    bounds = {"least" : [[], [], []],
               "random": [[]],
               "top2"  : [[], [], []]}

    
    for fname in file_list:
        nsamps, niters, true_id, true_label, target_label, img_info, activation, order = parse_filename(fname)

        # keys in mat:
        # ['Li_max', 'pred', 'G1_max', 'g_x0', 'path', 'info', 'G2_max', 'true_label', 'args', 'L1_max', 'Gi_max', 'L2_max', 'id', 'target_label']
        mat = sio.loadmat(fname)
        print('loading {}'.format(fname))
        
        if order == "1":
            G1_max = np.squeeze(mat['G1_max'])
            G2_max = np.squeeze(mat['G2_max'])
            Gi_max = np.squeeze(mat['Gi_max'])
        else:
            raise RuntimeError('!!! order is {}'.format(order))

        
        g_x0 = np.squeeze(mat['g_x0'])
        target_label = np.squeeze(mat['target_label'])
        true_id = np.squeeze(mat['id'])
        true_label = np.squeeze(mat['true_label'])
        img_info = mat['info'][0]
        
        # get the filename (.mat)
        print('[Filename] {}'.format(fname))       
        # get the model name (inception, cifar_2-layer)
        possible_names = ["mnist", "cifar", "mobilenet", "inception", "resnet"]
        model = dataset
        
        
        if order == "1":
            if norm == '1':
                Est_G = get_lipschitz_estimate(Gi_max, pool, "Li", True)
            elif norm == '2':
                Est_G = get_lipschitz_estimate(G2_max, pool, "L2", True)
            elif norm == 'i':
                Est_G = get_lipschitz_estimate(G1_max, pool, "L1", True)
                
        # the estimated Lipschitz constant
        Lip_G = Est_G['Lips_est']
        
        # compute robustness bound
        if order == "1": 
            bnd_L = g_x0 / Lip_G

        
        bounds[img_info][0].append(bnd_L)
        
        # original data_process mode
        #print('[STATS][L1] id = {}, true_label = {}, target_label = {}, info = {}, bnd_L1 = {:.5g}, bnd_L2 = {:.5g}, bnd_Li = {:.5g}'.format(true_id, true_label, target_label, img_info, bnd_L1, bnd_L2, bnd_Li))
        
        
        sys.stdout.flush()
        
    reduce_op = lambda x: sum(x) / len(x) if len(x) > 0 else 0


    # shutdown thread pool
    pool.close()
    pool.join()

    return reduce_op(bounds['random'][0]), time.time()-timestart
