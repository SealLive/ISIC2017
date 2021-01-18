import os
import copy
import torch
import torch.nn as nn
import numpy as np
from inception import inception_v3

class NetModel(object):
    """ Class for Network Model """
    def __init__(self, config):
        self.gpu = config.gpu
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.num_classes = config.num_classes
        self.checkpoint_path = config.checkpoint_path
        self.network_name = config.net.replace(" ", "_").lower()
        
        self._set_net_model()
        
    def _set_net_model(self):
        self.device = torch.device("cuda:{}".format(self.gpu))
        self.network = self._get_network().to(self.device)
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adadelta(self.network.parameters(), rho=0.95, eps=1e-8, lr=self.learning_rate)
        
    def _get_network(self):
        network = Classifier(self.num_classes)
        return network
    
    def train(self, data, label, to_softmax=True):
        # set network status
        self.network.train()
        self.network.zero_grad()
        # convert data and label to gpu
        _data = data.to(self.device)
        _label = label.to(self.device)
        _network_output = self.network(_data)
        # compute error and bp
        err = self.criterion(_network_output, _label)
        err.backward()
        self.optimizer.step()
        
        if to_softmax:
            return err.cpu().item(), torch.exp(_network_output.detach())
        else:
            return err.cpu().item(), _network_output.detach()
        
    def predict(self, data, label, to_softmax=True):
        # set network status
        self.network.eval()
        # convert data and label to gpu
        _data = data.to(self.device)
        _label = label.to(self.device)
        _network_output = self.network(_data)
        err = self.criterion(_network_output, _label)
        if to_softmax:
            return err.cpu().item(), torch.exp(_network_output.detach())
        else:
            return err.cpu().item(), _network_output.detach()
    
    def get_cost(self, label, output):
        label = label.to(self.device)
        return torch.nn.functional.nll_loss(torch.log(output), label, reduction="elementwise_mean").cpu().item()
    
    def is_correct(self, label, output):
        label = label.to(self.device)
        predicts = torch.argmax(output, dim=1)
        count = torch.sum(predicts==label)
        count = count.cpu().item()
        return count
    
    def is_correct_scalar(self, label, output):
        label = label.to(self.device)
        batch_size = output.shape[0]
        count = 0
        for b in range(batch_size):
            o = output[b, :]
            p = torch.argmax(o)
            if p == label[b]:
                count += 1
        return count
        
    def save_params(self, epoch, checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = "{}/{}_epoch_{}.pth".format(checkpoint_path, self.network_name, epoch)
        torch.save(self.network.state_dict(), save_path)
        print("Save network parameters to {}".format(save_path))
        
    def load_params(self, epoch, checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
            load_path = "{}/{}_epoch_{}.pth".format(checkpoint_path, self.network_name, epoch)
        else:
            load_path = checkpoint_path
        self.network.load_state_dict(torch.load(load_path, map_location=self.device))
        print("Load network parameters from {}".format(load_path))
        
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        m = inception_v3()
        
        tdct = torch.load("/root/workspace/wlt_pytorch/pre_trained_params/inception_v3_google-1a9a5a14.pth")
        remove_keys = []
        for key in tdct.keys():
            if "fc" in key:
                remove_keys.append(key)
        for key in remove_keys:
            tdct.__delitem__(key)
        m.load_state_dict(tdct, strict=False)
        self.features = m
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        out = self.features(x)
        out = self.log_softmax(out)
        return out