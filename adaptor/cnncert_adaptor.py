import torch
import torch.nn as nn
import numpy as np

import pytorch2keras

from adaptor.basic_adaptor import VerifierAdaptor
from datasets import get_input_shape

from tensorflow import keras
from tensorflow.keras import backend as K



class CNNCertBase(VerifierAdaptor):

    def __init__(self, dataset, model):
        super(CNNCertBase, self).__init__(dataset, model)

        input_shape = get_input_shape(dataset)
        k_model = pytorch2keras.pytorch_to_keras(
            self.model,
            torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0, 1, (1,) + input_shape))).cuda(),
            verbose=True
        )
        self.new_model = k_model

        print(self.new_model.summary())


class CNNCertAdaptor(CNNCertBase):

    pass

class FastLinSparseAdaptor(CNNCertBase):

    pass
