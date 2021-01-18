import copy
import torch
import numpy as np
import torch.nn as nn

class GVS(object):
    def __init__(self):
        # changing parameters
        self.gpu = "1"
        self.batch_size = 50
        self.max_max_epoch = 200
        self.exp_target = "ISIC"
        self.test_index = 3
        
        # fixing parameters
        self.net = "Inception V3"
        self.width = 299
        self.height = 299
        self.channel = 3
        self.num_classes = 2
        self.input_dim = (self.batch_size, self.channel, self.height, self.width)
        self.output_dim = (self.batch_size, self.num_classes)
        self.optimizer = "adadelta"
        self.learning_rate = 1.0
        self.keep_prob = 0.8
        self.wd = 1e-7
        self.checkpoint_path = "gpu_models"
        
#         self._get_data_list()
        
    def _get_data_list(self):
        train_index_list = []
        for i in range(1, 6):
            if i != self.test_index:
                train_index_list.append(i)
        self.train_index_list = copy.deepcopy(train_index_list)
        self.test_index_list = copy.deepcopy([self.test_index])