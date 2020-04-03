#Interface to run CW/EAD attacks

import tensorflow as tf
import numpy as np
from Attacks.li_attack import CarliniLi
from Attacks.l2_attack import CarliniL2
from Attacks.l1_attack import EADL1
from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_tinyimagenet import tinyImagenet
from tensorflow.contrib.keras.api.keras.models import load_model, Sequential
from utils import generate_data
from tensorflow.contrib.keras.api.keras import backend as K
import time as timer
import tensorflow as tf
import random
from train_resnet import *

def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

#Runs CW/EAD attack with specified norm
def cw_attack(file_name, norm, sess, num_image=10, cifar = False, tinyimagenet = False):
    np.random.seed(1215)
    tf.set_random_seed(1215)
    random.seed(1215)
    if norm == '1':
        attack = EADL1
        norm_fn = lambda x: np.sum(np.abs(x),axis=(1,2,3))
    elif norm == '2':
        attack = CarliniL2
        norm_fn = lambda x: np.sum(x**2,axis=(1,2,3))
    elif norm == 'i':
        attack = CarliniLi
        norm_fn = lambda x: np.max(np.abs(x),axis=(1,2,3))

    if cifar:
        data = CIFAR()
    elif tinyimagenet:
        data = tinyImagenet()
    else:
        data = MNIST()
    model = load_model(file_name, custom_objects={'fn':loss,'tf':tf, 'ResidualStart' : ResidualStart, 'ResidualStart2' : ResidualStart2})
    inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=num_image, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.predict, start=0)
    model.predict = model
    model.num_labels = 10
    if cifar:
        model.image_size = 32
        model.num_channels = 3
    elif tinyimagenet:
        model.image_size = 64
        model.num_channels = 3
        model.num_labels = 200
    else:
        model.image_size = 28
        model.num_channels = 1
        
    
    start_time = timer.time()
    attack = attack(sess, model, max_iterations = 1000)
    perturbed_input = attack.attack(inputs, targets)
    UB = np.average(norm_fn(perturbed_input-inputs))
    return UB, (timer.time()-start_time)/len(inputs)
    
