## l1_attack.py -- attack a network optimizing elastic-net distance with an l1 decision rule
##
## Copyright (C) 2017, Yash Sharma <ysharma1126@gmail.com>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
import time

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
BETA = 1e-3              # Hyperparameter trading off L2 minimization for L1 minimization

class EADL1:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 print_every = 100, early_stop_iters = 0,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST, beta = BETA):
        """
        EAD with L1 Decision Rule 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        print("early stop:", self.early_stop_iters)
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.beta = beta
        self.beta_t = tf.cast(self.beta, tf.float32)

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.newimg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.slack = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_newimg = tf.placeholder(tf.float32, shape)
        self.assign_slack = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_t = tf.cast(self.global_step, tf.float32)

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        self.zt = tf.divide(self.global_step_t, self.global_step_t+tf.cast(3, tf.float32))

        cond1 = tf.cast(tf.greater(tf.subtract(self.slack, self.timg),self.beta_t), tf.float32)
        cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(self.slack,self.timg)),self.beta_t), tf.float32)
        cond3 = tf.cast(tf.less(tf.subtract(self.slack, self.timg),tf.negative(self.beta_t)), tf.float32)

        upper = tf.minimum(tf.subtract(self.slack,self.beta_t), tf.cast(0.5, tf.float32))
        lower = tf.maximum(tf.add(self.slack,self.beta_t), tf.cast(-0.5, tf.float32))

        self.assign_newimg = tf.multiply(cond1,upper)+tf.multiply(cond2,self.timg)+tf.multiply(cond3,lower)
        self.assign_slack = self.assign_newimg+tf.multiply(self.zt, self.assign_newimg-self.newimg)
        self.setter = tf.assign(self.newimg, self.assign_newimg)
        self.setter_y = tf.assign(self.slack, self.assign_slack)
        """--------------------------------"""
        # prediction BEFORE-SOFTMAX of the model
        self.output = model.predict(self.newimg)
        self.output_y = model.predict(self.slack)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-self.timg),[1,2,3])
        self.l2dist_y = tf.reduce_sum(tf.square(self.slack-self.timg),[1,2,3])
        self.l1dist = tf.reduce_sum(tf.abs(self.newimg-self.timg),[1,2,3])
        self.l1dist_y = tf.reduce_sum(tf.abs(self.slack-self.timg),[1,2,3])
        self.elasticdist = self.l2dist + tf.multiply(self.l1dist, self.beta_t)
        self.elasticdist_y = self.l2dist_y + tf.multiply(self.l1dist_y, self.beta_t)
        
        # compute the probability of the label class versus the maximum other
        self.real = tf.reduce_sum((self.tlab)*self.output,1)
        self.real_y = tf.reduce_sum((self.tlab)*self.output_y,1)
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)
        self.other_y = tf.reduce_max((1-self.tlab)*self.output_y - (self.tlab*10000),1)
        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
            loss1_y = tf.maximum(0.0, self.other_y-self.real_y+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, self.real-self.other+self.CONFIDENCE)
            loss1_y = tf.maximum(0.0, self.real_y-self.other_y+self.CONFIDENCE)

        # sum up the losses
#         self.loss21 = tf.reduce_sum(self.l1dist)
# #        self.loss21_y = tf.reduce_sum(self.l1dist_y)
#         self.loss2 = tf.reduce_sum(self.l2dist)
#         self.loss2_y = tf.reduce_sum(self.l2dist_y)
#         self.loss1 = tf.reduce_sum(self.const*loss1)
#         self.loss1_y = tf.reduce_sum(self.const*loss1_y)

        self.loss21 = self.l1dist
#        self.loss21_y = tf.reduce_sum(self.l1dist_y)
        self.loss2 = self.l2dist
        self.loss2_y = self.l2dist_y
        self.loss1 = self.const*loss1
        self.loss1_y = self.const*loss1_y


        self.loss_opt = self.loss1_y+self.loss2_y
        self.loss = self.loss1+self.loss2+tf.multiply(self.beta_t,self.loss21)


        print("self.loss = ", self.loss)
        print("self.loss_opt = ", self.loss_opt)
        print("self.loss1_y = ", self.loss1_y)
        print("self.loss2_y = ", self.loss2_y)
        print("self.real = ", self.real)
        print("self.other = ", self.other)
        
        self.learning_rate = tf.train.polynomial_decay(self.LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5) 
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss_opt, var_list=[self.slack], global_step=self.global_step)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[self.global_step]+[self.slack]+[self.newimg]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the EAD attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            #r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size])[0])
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        print("="*10, "batch_size = ", batch_size, "="*10)

        # # convert to tanh-space
        # imgs = np.arctanh(imgs*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10
        n_success = 0

        # the best l2, score, and image attack
        o_bestl1 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        # fill the array as nan to indicate attack failure
        for b in o_bestattack:
            b.fill(np.nan)
        o_best_const = [self.initial_const]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print("current best l1", o_bestl1)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl1 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            self.sess.run(self.setter, feed_dict={self.assign_newimg: batch})
            self.sess.run(self.setter_y, feed_dict={self.assign_slack: batch})
            prev = 1e6
            train_timer = 0.0
            for iteration in range(self.MAX_ITERATIONS):
                # print out the losses every 10%
                # print("iteration = ", iteration)
                if iteration%(self.MAX_ITERATIONS//self.print_every) == 0:
                    # print(iteration,self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2)))
                    # grad = self.sess.run(self.grad_op)
                    # old_modifier = self.sess.run(self.modifier)
                    # np.save('white_iter_{}'.format(iteration), modifier)
                    loss, real, other, loss1, loss2, loss21 = self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2, self.loss21))
                    # print("loss = ", loss)
                    # print("real = ", real)
                    # print("other = ", other)
                    # print("loss1 = ", loss1)
                    # print("loss2 = ", loss2)

                    if self.batch_size == 1:
                        print("[STATS][L2] iter = {}, time = {:.3f}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, loss21 = {:.5g}".format(iteration, train_timer, loss[0], real[0], other[0], loss1[0], loss2[0], loss21[0]))
                        #print("[STATS][L2] iter = {}, time = {:.3f}, real = {:.5g}, other = {:.5g}".format(iteration, train_timer, real[0], other[0]))
                    elif self.batch_size > 10:
                        print("[STATS][L2][SUM of {}] iter = {}, time = {:.3f}, batch_size = {}, n_success = {:.5g}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, loss21 = {:.5g}".format(self.batch_size, iteration, train_timer, batch_size, n_success, sum(loss), sum(real), sum(other), sum(loss1), sum(loss2), sum(loss21)))
                        # print("[STATS][L2][SUM of {}] iter = {}, time = {:.3f}, batch_size = {}, n_success = {:.5g}, real = {:.5g}, other = {:.5g}".format(self.batch_size, iteration, train_timer, batch_size, n_success, sum(real), sum(other)))
                    else:
                        print("[STATS][L2] iter = {}, time = {:.3f}".format(iteration, train_timer))
                        print("[STATS][L2] real =", real)
                        print("[STATS][L2] other =", other)
                        print("[STATS][L2] loss1 =", loss1)
                        print("[STATS][L2] loss2 =", loss2)
                        print("[STATS][L2] loss21 =", loss21)
                        print("[STATS][L2] loss =", loss)
                    sys.stdout.flush()

                attack_begin_time = time.time()

                # perform the attack 
                self.sess.run([self.train])
                self.sess.run([self.setter, self.setter_y])
                l, l2s, l1s, elastic, scores, nimg = self.sess.run([self.loss, self.l2dist, self.l1dist, self.elasticdist, self.output, self.newimg])


                # print out the losses every 10%
                """
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2,self.loss21)))
                """
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                    # print("bookkeeping previous loss, iteration = ", iteration, ', l = ', l)
                    if np.all(l > prev*.9999):
                    #if l > prev*.9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev = l

                # adjust the best result found so far
                read_last_loss = False
                for e,(l1,sc,ii) in enumerate(zip(l1s,scores,nimg)):
                    if l1 < bestl1[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl1[e] = l1
                        bestscore[e] = np.argmax(sc)
                    if l1 < o_bestl1[e] and compare(sc, np.argmax(batchlab[e])):
                        # print a message if it is the first attack found
                        if o_bestl1[e] == 1e10:
                            if not read_last_loss: 
                                loss, real, other, loss1, loss2, loss21 = self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2, self.loss21))
                                read_last_loss = True
                            print("[STATS][L3][First valid attack found!] iter = {}, time = {:.3f}, img = {}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, loss21 = {:.5g}".format(iteration, train_timer, e, loss[e], real[e], other[e], loss1[e], loss2[e], loss21[e]))
                            #print("[STATS][L3][First valid attack found!] iter = {}, time = {:.3f}, img = {}, real = {:.5g}, other = {:.5g}".format(iteration, train_timer, e, real[e], other[e]))

                            n_success += 1
                        o_bestl1[e] = l1
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        o_best_const[e] = CONST[e]

                train_timer += time.time() - attack_begin_time

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # modifier = self.sess.run(self.modifier)
                    # np.save("best.model", modifier)
                    print('old constant: ', CONST[e])
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    print('new constant: ', CONST[e])
                else:
                    print('old constant: ', CONST[e])
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10
                    print('new constant: ', CONST[e])

        # return the best solution found
        o_bestl1 = np.array(o_bestl1)
        return np.array(o_bestattack), o_best_const
