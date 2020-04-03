#Simple CNNModel class

import os
import numpy as np
from PIL import Image
np.random.seed(99)
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
import time

class CNNModel:
    def __init__(self, model, inp_shape = (64,64,3)):
        self.model = model
        self.image_size = inp_shape[0]
        self.num_channels = inp_shape[2]

    def predict(self, data):
        return self.model(data)
