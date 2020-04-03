"""
setup_tinyimagenet.py

Tinyimagenet data and model loading code

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""

import os
import numpy as np
from PIL import Image


import time 

def load_images(trainPath):
    np.random.seed(1215)
    #Load images
    
    print('Loading 200 classes')

    num_classes = 200
    VAL_FRACTION = 0.1
    TEST_FRACTION = 0.1

    X_train=np.zeros([num_classes*500,3,64,64],dtype='uint8')
    y_train=np.zeros([num_classes*500], dtype='uint8')


    i=0
    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        annotations[sChild]=j
        for c in os.listdir(sChildPath):
            X=np.array(Image.open(os.path.join(sChildPath,c)))
            if len(np.shape(X))==2:
                X_train[i]=np.array([X,X,X])
            else:
                X_train[i]=np.transpose(X,(2,0,1))
            y_train[i]=j
            i+=1
        j+=1
        if (j >= num_classes):
            break
    
    num_pts = X_train.shape[0]
    idx = np.random.permutation(num_pts)
    
    test_idx = idx[:int(TEST_FRACTION*num_pts)]
    train_idxs = idx[int(TEST_FRACTION*num_pts):] 

    val_idx = train_idxs[:int(VAL_FRACTION*num_pts)]
    train_idx = train_idxs[int(VAL_FRACTION*num_pts):]
    
    X_test = X_train[test_idx]
    y_test = np.eye(200)[y_train[test_idx]]
    
    X_val = X_train[val_idx] 
    y_val = np.eye(200)[y_train[val_idx]]
    
    X_train = X_train[train_idx]
    y_train = np.eye(200)[y_train[train_idx]]

    X_train = np.float32(X_train)
    X_val = np.float32(X_val)
    X_test = np.float32(X_test)


    print(" ========= data type ============")
    print("data type = {}".format(X_test.dtype))

    return X_train,y_train, X_val, y_val, X_test, y_test


class tinyImagenet():
    def __init__(self):

        trainPath='data/tiny-imagenet-200/train'
        
        # X_train shape: num_train*3*64*64 
        X_train, y_train, X_val, y_val, X_test, y_test = load_images(trainPath) 

        # convetion is num_train*size*size*channel, e.g. MNIST: num*28*28*1
        self.train_data = np.swapaxes(X_train,1,3)
        self.train_labels = y_train

        self.validation_data = np.swapaxes(X_val,1,3)
        self.validation_labels = y_val
        
        self.test_data = np.swapaxes(X_test,1,3)
        self.test_labels = y_test
        
