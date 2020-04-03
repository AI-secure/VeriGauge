#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_gradient_norm.py

A multithreaded gradient norm sampler

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                 and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

from __future__ import division

import numpy as np
import random
import ctypes
import time
import sys
import os
import tensorflow as tf

from multiprocessing import Pool, current_process, cpu_count
from CLEVER.shmemarray import ShmemRawArray, NpShmemArray
from functools import partial
from CLEVER.randsphere import randsphere
from tensorflow.python.ops import gradients_impl
from tensorflow.contrib.keras.api.keras.models import load_model
from train_resnet import ResidualStart, ResidualStart2
from CLEVER.CNNModel import CNNModel

    
def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

class EstimateLipschitz(object):

    def __init__(self, sess, seed = 1215, nthreads = 0):
        """
        sess: tensorflow session
        Nsamp: number of samples to take per iteration
        Niters: number of iterations, each iteration we return a max L
        """
        self.sess = sess
        self.seed = seed
        # create a pool of workers to compute samples in advance
        if nthreads == 0:
            self.n_processes = max(cpu_count() // 2, 1)
        else:
            self.n_processes = nthreads
        # set up random seed during initialization
        def initializer(s):
            np.random.seed(s + current_process()._identity[0])
            # using only 1 OpenMP thread
            os.environ['OMP_NUM_THREADS'] = "1"
        self.pool = Pool(processes = self.n_processes, initializer = initializer, initargs=(self.seed,))

    def load_model(self, dataset = "mnist", model_name = "2-layer", activation = "relu", model = None, batch_size = 0, compute_slope = False, order = 1):
        """
        model: if set to None, then load dataset with model_name. Otherwise use the model directly.
        dataset: mnist, cifar and imagenet. recommend to use mnist and cifar as a starting point.
        model_name: possible options are 2-layer, distilled, and normal
        """
        from setup_cifar import CIFAR, CIFARModel, TwoLayerCIFARModel
        from setup_mnist import MNIST, MNISTModel, TwoLayerMNISTModel
        from setup_tinyimagenet import tinyImagenet

        # if set this to true, we will use the logit layer output instead of probability
        # the logit layer's gradients are usually larger and more stable
        output_logits = True
        self.dataset = dataset
        self.model_name = model_name

        if model is None:
            print('Loading model...')
            if dataset == "mnist":
                self.batch_size = 1024
                
                model = CNNModel(load_model(model_name, custom_objects={'fn':fn, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf}), inp_shape = (28,28,1))
            elif dataset == "cifar":
                self.batch_size = 1024

                model = CNNModel(load_model(model_name, custom_objects={'fn':fn, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf}), inp_shape = (32,32,3))
            elif dataset == "tinyimagenet":
                self.batch_size = 32

                model = CNNModel(load_model(model_name, custom_objects={'fn':fn, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf}), inp_shape = (64,64,3))
            else:
                raise(RuntimeError("dataset unknown"))

        #print("*** Loaded model successfully")

        self.model = model
        self.compute_slope = compute_slope
        if batch_size != 0:
            self.batch_size = batch_size
        
        ## placeholders: self.img, self.true_label, self.target_label
        # img is the placeholder for image input
        self.img = tf.placeholder(shape = [None, model.image_size, model.image_size, model.num_channels], dtype = tf.float32)
        # output is the output tensor of the entire network
        self.output = model.predict(self.img)
        # create the graph to compute gradient
        # get the desired true label and target label
        self.true_label = tf.placeholder(dtype = tf.int32, shape = [])
        self.target_label = tf.placeholder(dtype = tf.int32, shape = [])
        true_output = self.output[:, self.true_label]
        target_output = self.output[:, self.target_label]
        # get the difference
        self.objective = true_output - target_output
        # get the gradient(deprecated arguments)
        self.grad_op = tf.gradients(self.objective, self.img)[0]
        # compute gradient norm: (in computation graph, so is faster)
        grad_op_rs = tf.reshape(self.grad_op, (tf.shape(self.grad_op)[0], -1))
        self.grad_2_norm_op = tf.norm(grad_op_rs, axis = 1)
        self.grad_1_norm_op = tf.norm(grad_op_rs, ord=1, axis = 1)
        self.grad_inf_norm_op = tf.norm(grad_op_rs, ord=np.inf, axis = 1)
        
        ### Lily: added Hessian-vector product calculation here for 2nd order bound:
        if order == 2:
            ## _hessian_vector_product(ys, xs, v): return a list of tensors containing the product between the Hessian and v
            ## ys: a scalar valur or a tensor or a list of tensors to be summed to yield of scalar
            ## xs: a list of tensors that we should construct the Hessian over
            ## v: a list of tensors with the same shape as xs that we want to multiply by the Hessian
            # self.randv: shape = (Nimg,28,28,1) (the v in _hessian_vector_product)
            self.randv = tf.placeholder(shape = [None, model.image_size, model.image_size, model.num_channels], dtype = tf.float32)
            # hv_op_tmp: shape = (Nimg,28,28,1) for mnist, same as self.img (the xs in _hessian_vector_product)
            hv_op_tmp = gradients_impl._hessian_vector_product(self.objective, [self.img], [self.randv])[0]
            # hv_op_rs: reshape hv_op_tmp to hv_op_rs whose shape = (Nimg, 784) for mnist
            hv_op_rs = tf.reshape(hv_op_tmp, (tf.shape(hv_op_tmp)[0],-1))
            # self.hv_norm_op: norm of hessian vector product, keep shape = (Nimg,1) using keepdims
            self.hv_norm_op = tf.norm(hv_op_rs, axis = 1, keepdims=True)
            # hv_op_rs_normalize: normalize Hv to Hv/||Hv||, shape = (Nimg, 784)
            hv_op_rs_normalize = hv_op_rs/self.hv_norm_op
            # self.hv_op: reshape hv_op_rs_normalize to shape = (Nimg,28,28,1)
            self.hv_op = tf.reshape(hv_op_rs_normalize, tf.shape(hv_op_tmp))
            
            ## reshape randv and compute its norm
            # shape: (Nimg, 784)
            randv_rs = tf.reshape(self.randv, (tf.shape(self.randv)[0],-1))
            # shape: (Nimg,)
            self.randv_norm_op = tf.norm(randv_rs, axis = 1)
            ## compute v'Hv: use un-normalized Hv (hv_op_tmp, hv_op_rs)
            # element-wise multiplication and then sum over axis = 1 (now shape: (Nimg,)) 
            self.vhv_op = tf.reduce_sum(tf.multiply(randv_rs,hv_op_rs),axis=1)
            ## compute Rayleigh quotient: v'Hv/v'v (estimated largest eigenvalue), shape: (Nimg,)
            # note: self.vhv_op and self.randv_norm_op has to be in the same dimension (either (Nimg,) or (Nimg,1))
            self.eig_est = self.vhv_op/tf.square(self.randv_norm_op) 
            
            ## Lily added the tf.while to compute the eigenvalue in computational graph later
            # cond for computing largest abs/neg eigen-value
            def cond(it, randv, eig_est, eig_est_prev, tfconst):
                norm_diff = tf.norm(eig_est-eig_est_prev,axis=0)
                return tf.logical_and(it < 500, norm_diff > 0.001)

            # compute largest abs eigenvalue: tfconst = 0
            # compute largest neg eigenvalue: tfconst = 10 
            def body(it, randv, eig_est, eig_est_prev, tfconst):
                #hv_op_tmp = gradients_impl._hessian_vector_product(self.objective, [self.img], [randv])[0]-10*randv
                hv_op_tmp = gradients_impl._hessian_vector_product(self.objective, [self.img], [randv])[0]-tf.multiply(tfconst,randv)
                hv_op_rs = tf.reshape(hv_op_tmp, (tf.shape(hv_op_tmp)[0],-1))
                hv_norm_op = tf.norm(hv_op_rs, axis = 1, keepdims=True)
                hv_op_rs_normalize = hv_op_rs/hv_norm_op
                hv_op = tf.reshape(hv_op_rs_normalize, tf.shape(hv_op_tmp))
                
                randv_rs = tf.reshape(randv, (tf.shape(randv)[0],-1))
                randv_norm_op = tf.norm(randv_rs, axis = 1)
                vhv_op = tf.reduce_sum(tf.multiply(randv_rs,hv_op_rs),axis=1)
                eig_est_prev = eig_est
                eig_est = vhv_op/tf.square(randv_norm_op) 
                
                return (it+1, hv_op, eig_est, eig_est_prev, tfconst)

            it = tf.constant(0)
            # compute largest abs eigenvalue
            result = tf.while_loop(cond, body, [it, self.randv, self.vhv_op, self.eig_est, tf.constant(0.0)])
            # compute largest neg eigenvalue
            self.shiftconst = tf.placeholder(shape = (), dtype = tf.float32)
            result_1 = tf.while_loop(cond, body, [it, self.randv, self.vhv_op, self.eig_est, self.shiftconst])

            # computing largest abs eig value and save result
            self.it = result[0]
            self.while_hv_op = result[1]
            self.while_eig = result[2]
            
            # computing largest neg eig value and save result
            self.it_1 = result_1[0]
            #self.while_eig_1 = tf.add(result_1[2], tfconst)
            self.while_eig_1 = tf.add(result_1[2], result_1[4])

            show_tensor_op = False
            if show_tensor_op: 
                print("====================")
                print("Define hessian_vector_product operator: ")
                print("hv_op_tmp = {}".format(hv_op_tmp))
                print("hv_op_rs = {}".format(hv_op_rs))
                print("self.hv_norm_op = {}".format(self.hv_norm_op))
                print("hv_op_rs_normalize = {}".format(hv_op_rs_normalize))
                print("self.hv_op = {}".format(self.hv_op))
                print("self.grad_op = {}".format(self.grad_op))
                print("randv_rs = {}".format(randv_rs))
                print("self.randv_norm_op = {}".format(self.randv_norm_op))
                print("self.vhv_op = {}".format(self.vhv_op))
                print("self.eig_est = {}".format(self.eig_est))
                print("====================")

        return self.img, self.output, self.model


    def _estimate_Lipschitz_multiplerun(self, num, niters, input_image, target_label, true_label, sample_norm = "l2", transform=None, order = 1):
        """
        num: number of samples per iteration
        niters: number of iterations
        input_image: original image (h*w*c)
        """
        batch_size = self.batch_size
        shape = (batch_size, self.model.image_size, self.model.image_size, self.model.num_channels)
        dimension = self.model.image_size * self.model.image_size * self.model.num_channels

        if num < batch_size:
            print("Increasing num to", batch_size)
            num = batch_size
        
        """
        1. Compute input_image related quantities:
        """
        # get the original prediction and gradient, gradient norms values on input image:
        pred, grad_val, grad_2_norm_val, grad_1_norm_val, grad_inf_norm_val = self.sess.run(
          [self.output, self.grad_op, self.grad_2_norm_op, self.grad_1_norm_op, self.grad_inf_norm_op], 
          feed_dict = {self.img: [input_image], self.true_label: true_label, self.target_label: target_label})
        pred = np.squeeze(pred)
        # print(pred)
        # print(grad_val)
        
        # class c and class j in Hein's paper. c is original class
        c = true_label
        j = target_label
        # get g_x0 = f_c(x_0) - f_j(x_0)
        g_x0 = pred[c] - pred[j]
        # grad_z_norm should be scalar
        g_x0_grad_2_norm = np.squeeze(grad_2_norm_val)
        g_x0_grad_1_norm = np.squeeze(grad_1_norm_val)
        g_x0_grad_inf_norm = np.squeeze(grad_inf_norm_val)

        print("** Evaluating g_x0, grad_2_norm_val on the input image x0: ")
        print("shape of input_image = {}".format(input_image.shape))
        print("g_x0 = {:.3f}, grad_2_norm_val = {:3f}, grad_1_norm_val = {:.3f}, grad_inf_norm_val = {:3f}".format(g_x0, g_x0_grad_2_norm, g_x0_grad_1_norm, g_x0_grad_inf_norm))

        ##### Lily #####
        if order == 2: # evaluate the hv and hv norm on input_image
            # set randv as a random matrix with the same shape as input_image
            print("** Evaluating hv and hv_norm on the input image x0:")
            randv = np.random.randn(*input_image.shape)
            hv, hv_norm = self.sess.run([self.hv_op, self.hv_norm_op], 
                    feed_dict = {self.img: [input_image], self.randv:[randv], self.true_label: true_label, self.target_label: target_label}) 
            print("hv shape = {}, hv_norm = {}".format(hv.shape, hv_norm))


        """
        2. Prepare for sampling: 
        """
        def div_work_to_cores(njobs, nprocs):
            process_item_list = []
            while njobs > 0:
                process_item_list.append(int(np.ceil(njobs / float(nprocs))))
                njobs -= process_item_list[-1]
                nprocs -= 1
            return process_item_list
        # n is the dimension

        if self.dataset == "imagenett":
            # for imagenet, generate random samples for this batch only
            # array in shared memory storing results of all threads
            total_item_size = batch_size
        else:
            # for cifar and mnist, generate random samples for this entire iteration
            total_item_size = num
        # divide the jobs evenly to all available threads
        process_item_list = div_work_to_cores(total_item_size, self.n_processes)
        self.n_processes = len(process_item_list)
        # select random sample generation function
        if sample_norm == "l2":
            # the scaling constant in [a,b]: scale the L2 norm of each sample (has originally norm ~1)
            a = 0; b = 3; 
        elif sample_norm == "li":
            # for Linf we don't need the scaling
            a = 0.1; b = 0.1; 
        elif sample_norm == "l1":
            # TODO: make the sample ball radius adjustable
            a = 0; b = 30;
        else:
            raise RuntimeError("Unknown sample_norm " + sample_norm)
        print('Using sphere', sample_norm)
        
        ## create necessary shared array structures (saved in /dev/shm) and will be used (and written) in randsphere.py: 
        #   result_arr, scale, input_example, all_inputs
        #   note: need to use scale[:] = ... not scale = ..., o.w. the contents will not be saved to the shared array
        # inputs_0 is the image x_0
        inputs_0 = np.array(input_image)
        tag_prefix = str(os.getpid()) + "_"
        result_arr = NpShmemArray(np.float32, (total_item_size, dimension), tag_prefix + "randsphere")
        # we have an extra batch_size to avoid overflow
        scale = NpShmemArray(np.float32, (num+batch_size), tag_prefix + "scale")
        scale[:] = (b-a)*np.random.rand(num+batch_size)+a; 
        input_example = NpShmemArray(np.float32, inputs_0.shape, tag_prefix + "input_example")
        # this is a read-only array
        input_example[:] = inputs_0
        # all_inputs is a shared memeory array and will be written in the randsphere to save the samples 
        # all_inputs holds the perturbations for one batch or all samples
        all_inputs = NpShmemArray(np.float32, (total_item_size,) + inputs_0.shape, tag_prefix + "all_inputs")
        # holds the results copied from all_inputs
        clipped_all_inputs = np.empty(dtype=np.float32, shape = (total_item_size,) + inputs_0.shape)
        # prepare the argument list
        offset_list = [0]
        for item in process_item_list[:-1]:
            offset_list.append(offset_list[-1] + item)
        print(self.n_processes, "threads launched with parameter", process_item_list, offset_list)

        ## create multiple process to generate samples
        # randsphere: generate samples (see randsphere.py); partial is a function similar to lambda, now worker_func is a function of idx only
        worker_func = partial(randsphere, n = dimension, input_shape = inputs_0.shape, total_size = total_item_size, scale_size = num+batch_size, tag_prefix = tag_prefix, r = 1.0, norm = sample_norm, transform = transform)
        worker_args = list(zip(process_item_list, offset_list, [0] * self.n_processes))
        # sample_results is an object to monitor if the process has ended (meaning finish generating samples in randsphere.py)
        # this line of code will initiate the worker_func to start working (like initiate the job)
        sample_results = self.pool.map_async(worker_func, worker_args)

        # num: # of samples to be run, \leq samples.shape[0]
        
        # number of iterations
        Niters = niters;
        
        if order == 1:
            # store the max L in each iteration
            L2_max = np.zeros(Niters)
            L1_max = np.zeros(Niters)
            Li_max = np.zeros(Niters)
            # store the max G in each iteration
            G2_max = np.zeros(Niters)
            G1_max = np.zeros(Niters)
            Gi_max = np.zeros(Niters)
            # store computed Lispschitz constants in each iteration
            L2 = np.zeros(num)
            L1 = np.zeros(num)
            Li = np.zeros(num)
            # store computed gradient norm in each iteration
            G2 = np.zeros(num)
            G1 = np.zeros(num)
            Gi = np.zeros(num)
        elif order == 2:
            # store the max H in each iteration
            H2_max = np.zeros(Niters)
            # store computed 2 norm of H in each iteration 
            H2 = np.zeros(num)
            H2_neg = np.zeros(num)
        
        # how many batches we have
        Nbatches = num // batch_size

        # timer
        search_begin_time = time.time()

        """
        3. Start performing sampling:
        """
        ## Start 
        # multiple runs: generating the samples 
        ## use worker_func to generate x samples, and then use sess.run to evaluate the gradient norm operator
        for iters in range(Niters):
            iter_begin_time = time.time()
            
            # shuffled index 
            # idx_shuffle = np.random.permutation(num);
            
            # the scaling constant in [a,b]: scale the L2 norm of each sample (has originally norm ~1)
            scale[:] = (b-a)*np.random.rand(num+batch_size)+a; 

            # number of L's we have computed
            L_counter = 0
            G_counter = 0
            H_counter = 0

            overhead_time = 0.0
            overhead_start = time.time()
            # for cifar and mnist, generate all the random input samples (x in the paper) at once
            # for imagenet, generate one batch of input samples (x in the paper) for each iteration 
            if self.dataset != "imagenett": 
                # get samples for this iteration: make sure randsphere finished computing samples and stored in all_inputs 
                # if the samples have not yet done generating, then this line will block the codes until the processes are done, then it will return
                sample_results.get()
                # copy the results to a buffer and do clipping
                np.clip(all_inputs, 0, 255, out = clipped_all_inputs)
                # create multiple process again to generate samples for next batch (initiate a new job) because in below we will need to do sess.run in GPU which might be slow. So we want to generate samples on CPU while running sess.run on GPU to save time   
                sample_results = self.pool.map_async(worker_func, worker_args)
            overhead_time += time.time() - overhead_start
            
            ## generate input samples "batch_inputs" and compute corresponding gradient norms samples "perturbed_grad_x_norm"
            for i in range(Nbatches):
                overhead_start = time.time()
                # for imagenet, generate random samples for this batch only
                if self.dataset == "imagenett":
                    # get samples for this batch
                    sample_results.get()
                    # copy the results to a buffer and do clipping
                    np.clip(all_inputs, 0, 255, out = clipped_all_inputs)
                    # create multiple threads to generate samples for next batch
                    worker_args = zip(process_item_list, offset_list, [(i + 1) * batch_size] * self.n_processes)
                    sample_results = self.pool.map_async(worker_func, worker_args)

                if self.dataset == "imagenett":
                    # we generate samples for each batch at a time
                    batch_inputs = clipped_all_inputs
                else:
                    # we generate samples for all batches
                    batch_inputs = clipped_all_inputs[i * batch_size: (i + 1) * batch_size]
                # print(result_arr.shape, result_arr)
                # print('------------------------')
                # print(batch_inputs.shape, batch_inputs.reshape(result_arr.shape))
                # print('------------------------')
                overhead_time += time.time() - overhead_start
                
                if order == 1:
                    # run inference and get the gradient
                    perturbed_predicts, perturbed_grad_2_norm, perturbed_grad_1_norm, perturbed_grad_inf_norm = self.sess.run(
                            [self.output, self.grad_2_norm_op, self.grad_1_norm_op, self.grad_inf_norm_op], 
                            feed_dict = {self.img: batch_inputs, self.target_label: target_label, self.true_label: true_label})
                
                    if self.compute_slope:
                        # compute distance between consecutive samples: not use sequential samples 
                        s12_2_norm = np.linalg.norm(s[0:batch_size-1:2] - s[1:batch_size:2], axis = 1)
                        s12_1_norm = np.linalg.norm(s[0:batch_size-1:2] - s[1:batch_size:2], ord=1, axis = 1)
                        s12_i_norm = np.linalg.norm(s[0:batch_size-1:2] - s[1:batch_size:2], ord=np.inf, axis = 1)
                        # compute function value differences: not use sequential samples 
                        g_x1 = perturbed_predicts[0:batch_size-1:2, c] - perturbed_predicts[0:batch_size-1:2, j]
                        g_x2 = perturbed_predicts[1:batch_size:2, c] - perturbed_predicts[1:batch_size:2, j]
                        # estimated Lipschitz constants for this batch
                        # for slope estimate, we need the DUAL norm
                        batch_L2 = np.abs(g_x1 - g_x2) / s12_2_norm
                        batch_L1 = np.abs(g_x1 - g_x2) / s12_i_norm
                        batch_Li = np.abs(g_x1 - g_x2) / s12_1_norm
                        L2[L_counter : L_counter + batch_size//2] = batch_L2
                        L1[L_counter : L_counter + batch_size//2] = batch_L1
                        Li[L_counter : L_counter + batch_size//2] = batch_Li
                    
                    G2[G_counter : G_counter + batch_size] = perturbed_grad_2_norm
                    G1[G_counter : G_counter + batch_size] = perturbed_grad_1_norm
                    Gi[G_counter : G_counter + batch_size] = perturbed_grad_inf_norm
                    L_counter += (batch_size//2)
                    G_counter += batch_size
            
                elif order == 2: 
                ##### Lily #####
                    randv_batch = np.random.randn(*batch_inputs.shape)
                    perturbed_hv, perturbed_hv_norm = self.sess.run([self.hv_op, self.hv_norm_op],
                                                        feed_dict = {self.img: batch_inputs, self.randv: randv_batch, 
                                                                    self.true_label: true_label, self.target_label: target_label}) 
                    
                    show_tensor_dim = False                                    
                    if show_tensor_dim:
                        print("====================")
                        print("** Evaluating perturbed_hv and perturbed_hv_norm in batch {}: ".format(iters))
                        print("pertubed_hv_prod shape = {}".format(perturbed_hv.shape))
                        print("randv_batch shape = {}".format(randv_batch.shape))
                        print("perturbed_hv_norm = {}".format(perturbed_hv_norm[:,0])) # size: (Nimg, 1)
                        print("perturbed_hv_norm shape = {}".format(perturbed_hv_norm.shape))
                        #print("perturbed_grad_2_norm= {}".format(perturbed_grad_2_norm))
                        #print("perturbed_grad_2_norm shape = {}".format(perturbed_grad_2_norm.shape))
                    
            
                    pt_hvs = []
                    pt_hvs.append(perturbed_hv+0*randv_batch)
                    
                    #print("************** Using tf.while_loop:********************")
                    # compute max eigenvalue
                    temp_hv, temp_eig, niter_eig = self.sess.run([self.while_hv_op, self.while_eig, self.it], feed_dict = {self.img: batch_inputs, self.randv: randv_batch, self.true_label: true_label, self.target_label: target_label})
                    ##print("converge in {} steps, temp_eig = {}".format(niter_eig, temp_eig))

                    # if max eigenvalue is positive, compute the max neg eigenvalue by using the shiftconst 
                    if max(temp_eig) > 0:  
                        shiftconst = max(temp_eig)
                        temp_eig_1, niter_eig_1 = self.sess.run([self.while_eig_1, self.it_1], feed_dict = {self.img: batch_inputs, self.randv: randv_batch, self.true_label: true_label, self.target_label: target_label, self.shiftconst: shiftconst})
                        ##print("converge in {} steps, temp_eig_1 = {}".format(niter_eig_1, temp_eig_1))
                    else:
                        temp_eig_1 = temp_eig
                        niter_eig_1 = -1

                    print("temp_eig (abs) converge in {} steps, temp_eig_1 (neg) converge in {} steps".format(niter_eig, niter_eig_1))

                    ## use outer while_loop
                    #max_eig_iters = 10
                    #print_flag = True
                    #final_est_eig_1 = self._compute_max_abseig(pt_hvs, batch_inputs, true_label, target_label, max_eig_iters, print_flag)
                    #print("************** Using outer while_loop:********************")
                    #print("outer loop final_est_eig_1 = {}".format(final_est_eig_1))
                    
                    ## use tf while_loop
                    final_est_eig = temp_eig
                    final_est_eig_neg = temp_eig_1

                    H2[H_counter : H_counter + batch_size] = final_est_eig
                    H2_neg[H_counter : H_counter + batch_size] = final_est_eig_neg
                    H_counter += batch_size

            if order == 1:
                # at the end of each iteration: get the per-iteration max gradient norm
                if self.compute_slope:
                    L2_max[iters] = np.max(L2)
                    L1_max[iters] = np.max(L1)
                    Li_max[iters] = np.max(Li)
                G2_max[iters] = np.max(G2)
                G1_max[iters] = np.max(G1)
                Gi_max[iters] = np.max(Gi)
                
                
                if self.compute_slope:
                    print('[STATS][L2] loop = {}, time = {:.5g}, iter_time = {:.5g}, overhead = {:.5g}, L2 = {:.5g}, L1 = {:.5g}, Linf = {:.5g}, G2 = {:.5g}, G1 = {:.5g}, Ginf = {:.5g}'.format(iters, time.time() - search_begin_time, time.time() - iter_begin_time, overhead_time, L2_max[iters], L1_max[iters], Li_max[iters], G2_max[iters], G1_max[iters], Gi_max[iters]))
                else:
                    print('[STATS][L2] loop = {}, time = {:.5g}, iter_time = {:.5g}, overhead = {:.5g}, G2 = {:.5g}, G1 = {:.5g}, Ginf = {:.5g}'.format(iters, time.time() - search_begin_time, time.time() - iter_begin_time, overhead_time, G2_max[iters], G1_max[iters], Gi_max[iters]))
                sys.stdout.flush()
                # reset per iteration L and G by filling 0
                if self.compute_slope:
                    L2.fill(0)
                    L1.fill(0)
                    Li.fill(0)
                G2.fill(0)
                G1.fill(0)
                Gi.fill(0)
            elif order == 2:
                ## consider -lambda_min  
                idx = H2 > 0
                H2[idx] = H2_neg[idx]
                idx_max = np.argmax(abs(H2))
                H2_max[iters] = H2[idx_max]

                print('[STATS][L2] loop = {}, time = {:.5g}, iter_time = {:.5g}, overhead = {:.5g}, H2 = {:.5g}'.format(iters, time.time() - search_begin_time, time.time() - iter_begin_time, overhead_time, H2_max[iters]))

        if order == 1:
            print('[STATS][L1] g_x0 = {:.5g}, L2_max = {:.5g}, L1_max = {:.5g}, Linf_max = {:.5g}, G2_max = {:.5g}, G1_max = {:.5g}, Ginf_max = {:.5g}'.format(
                   g_x0, np.max(L2_max), np.max(L1_max), np.max(Li_max), np.max(G2_max), np.max(G1_max), np.max(Gi_max)))
            # when compute the bound we need the DUAL norm
            if self.compute_slope:
                print('[STATS][L1] bnd_L2_max = {:.5g}, bnd_L1_max = {:.5g}, bnd_Linf_max = {:.5g}, bnd_G2_max = {:.5g}, bnd_G1_max = {:.5g}, bnd_Ginf_max = {:.5g}'.format(g_x0/np.max(L2_max), g_x0/np.max(Li_max), g_x0/np.max(L1_max), g_x0/np.max(G2_max), g_x0/np.max(Gi_max), g_x0/np.max(G1_max)))
            else:
                print('[STATS][L1] bnd_G2_max = {:.5g}, bnd_G1_max = {:.5g}, bnd_Ginf_max = {:.5g}'.format(g_x0/np.max(G2_max), g_x0/np.max(Gi_max), g_x0/np.max(G1_max)))
            
            sys.stdout.flush()

            # discard the last batch of samples
            sample_results.get()
            return [L2_max,L1_max,Li_max,G2_max,G1_max,Gi_max,g_x0,pred]
        
        elif order == 2:
            # find positive eig value and substitute with its corresponding negative eig value, then we only need to sort once

            #print("H2_max = {}".format(H2_max))
            # find max abs(H2_max)
            H2_max_val = max(abs(H2_max))

            print('[STATS][L1] g_x0 = {:.5g}, g_x0_grad_2_norm = {:.5g}, g_x0_grad_1_norm = {:.5g}, g_x0_grad_inf_norm = {:.5g}, H2_max = {:.5g}'.format(g_x0, g_x0_grad_2_norm, g_x0_grad_1_norm, g_x0_grad_inf_norm, H2_max_val))
            
            bnd = (-g_x0_grad_2_norm + np.sqrt(g_x0_grad_2_norm**2+2*g_x0*H2_max_val))/H2_max_val
            print('[STATS][L1] bnd_H2_max = {:.5g}'.format(bnd))
            sys.stdout.flush()

            sample_results.get()
            return [H2_max, g_x0, g_x0_grad_2_norm, g_x0_grad_1_norm, g_x0_grad_inf_norm, pred]

    
    def _compute_max_abseig(self, pt_hvs, batch_inputs, true_label, target_label, max_eig_iters, print_flag):
        
        ## compute hv and est_eig:
        i = 0
        cond = False

        pt_eigs = [] 

        print("pt_hvs[0] shape = {}".format(pt_hvs[0].shape))

        # perform power iteration loop outside tensorflow
        while (i<max_eig_iters and cond==False):
            tmp_hv, tmp_hv_norm, tmp_vhv, tmp_vnorm, tmp_est_eig  = self.sess.run([self.hv_op, self.hv_norm_op, self.vhv_op, self.randv_norm_op, self.eig_est], feed_dict = {self.img: batch_inputs, self.randv: pt_hvs[i], self.true_label: true_label, self.target_label: target_label})
            tmp_vhv = np.squeeze(tmp_vhv)
            tmp_vnorm = np.squeeze(tmp_vnorm)
            tmp_est_eig = np.squeeze(tmp_est_eig)
            
            if print_flag:
                #print("current step = {}, norm = {}".format(i, tmp_hv_norm[:,0]))
                #print("current step = {}, pt_hv_prod.shape = {}, pt_hvs_norm.shape = {}".format(i,tmp_hv.shape, tmp_hv_norm.shape))
                print("current step = {}, est_eig = {}".format(i,tmp_est_eig-0))
                #print("current step = {}, vhv = {}".format(i,tmp_vhv))
                #print("current step = {}, vnorm (check: should be 1) = {}".format(i,tmp_vnorm))
            pt_hvs.append(tmp_hv+0*pt_hvs[i])
            pt_eigs.append(tmp_est_eig)

            # conditions
            if i > 0:
                cond_element = abs(tmp_est_eig-pt_eigs[i-1]) < 1e-3
                if print_flag:
                    print("cond = {}".format(cond_element))
                cond = cond_element.all()
            i+=1
            
            if i == max_eig_iters:
                print("==== Reach max iterations!!! ====")
        
        return pt_eigs[-1]


    def __del__(self):
        # terminate the pool
        self.pool.terminate()

    def estimate(self, x_0, true_label, target_label, Nsamp, Niters, sample_norm, transform, order):
        result = self._estimate_Lipschitz_multiplerun(Nsamp,Niters,x_0,target_label,true_label,sample_norm, transform, order)
        return result


