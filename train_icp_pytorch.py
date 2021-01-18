import torch
import torch.nn as nn
from global_variables import GVS
from seal_datasets import get_filename_list, MalignantFolder
import torchvision.transforms as transforms
from net_model import NetModel
import datetime, time, os
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn import metrics

# -------- Setting logger -------- #
# Time zone is set to Asia/BeiJing(which is UTC-8)
# logger.info() outputs logs to both console and file
# logger.debug() outputs logs only to console
tmp_gvs = GVS()
log_filename_prefix = "Train_" + tmp_gvs.exp_target + "_Test" + str(tmp_gvs.test_index) + "_" + tmp_gvs.net.replace(" ", "_") + "_"
os.environ['TZ'] = 'UTC-8'
time.tzset()
date_and_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log_filename = "Logs/" + log_filename_prefix + date_and_time + ".log"
if not os.path.exists("Logs"): os.makedirs("Logs")
if os.path.exists(log_filename): os.remove(log_filename)

logger = logging.getLogger()
fh = logging.FileHandler(log_filename)
ch = logging.StreamHandler()

fh.setLevel(logging.INFO)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %A %H:%M:%S")
fh.setFormatter(formatter)  
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
# -------------------------------- #

def main():
    train_gvs, test_gvs, train_loader, test_loader = load_data()
    model = NetModel(train_gvs)
    
    max_valid_acc = -1.0
    max_valid_auc = -1.0
    max_valid_acc_epoch = -1
    max_valid_auc_epoch = -1
    min_valid_fnr = 2.0
    min_valid_fnr_epoch = -1
    for epoch in range(train_gvs.max_max_epoch):
        logger.info("Epoch:%4d, learning rate = %.4f" % (epoch+1, model.optimizer.defaults["lr"]))
        train_acc, train_cm, train_loss, train_auc = train(train_gvs, model, train_loader, epoch+1)
        valid_acc, valid_cm, valid_loss, valid_auc = valid(test_gvs, model, test_loader, epoch+1)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            max_valid_acc_epoch = epoch + 1
        if valid_auc > max_valid_auc:
            max_valid_auc = valid_auc
            max_valid_auc_epoch = epoch + 1
            
        train_confusion_matrix = GuoConfusion(train_cm)
        valid_confusion_matrix = GuoConfusion(valid_cm)
        logger.info("Epoch:%4d, Train Accuracy: %.4f, AUC: %.4f" % (epoch+1, train_acc, train_auc))
        logger.info("Epoch:%4d, Valid Accuracy: %.4f, AUC: %.4f" % (epoch+1, valid_acc, valid_auc))
        logger.info("Epoch:%4d, Max valid accuracy is: %.4f at Epoch %d" % (epoch+1, max_valid_acc, max_valid_acc_epoch))
        logger.info("Epoch:%4d, Max valid AUC is: %.4f at Epoch %d" % (epoch+1, max_valid_auc, max_valid_auc_epoch))
        tp, tn, fp, fn = train_confusion_matrix.paper_log()
        log_str = "[Train] Epoch:%d, TP:%d, TN:%d, FP:%d, FN:%d, Loss:%.6f" % (epoch+1, tp, tn, fp, fn, train_loss)
        logger.info(log_str)
        tp, tn, fp, fn = valid_confusion_matrix.paper_log()
        log_str = "[Valid] Epoch:%d, TP:%d, TN:%d, FP:%d, FN:%d, Loss:%.6f" % (epoch+1, tp, tn, fp, fn, valid_loss)
        logger.info(log_str)
        valid_fnr = fn / (tp + fn)
        if valid_fnr < min_valid_fnr:
            min_valid_fnr = valid_fnr
            min_valid_fnr_epoch = epoch + 1
        log_str = "[Valid] Epoch:%d, False Negative Rate:%.6f, min valid fnr is %.6f at epoch %d" % (epoch+1, valid_fnr, min_valid_fnr, min_valid_fnr_epoch)
        logger.info(log_str)
        model.save_params(epoch+1)
    
def train(gvs, model, data_loader, epoch):
    batch_size = gvs.batch_size
    all_cost = []
    count = 0
    start_time = time.time()
    y_predict = []
    y_truth = []
    for sidx, [data, label, image_path] in enumerate(data_loader, 0):
        error, model_output = model.train(data, label)
        # ----- collect predicts and labels ----- #
        y_predict.append(model_output.cpu().numpy()[:, 1])
        y_truth.append(label.numpy())
        # --------------------------------------- #
        all_cost.append(error)
        count += model.is_correct(label, model_output)
        
        if (sidx+1) % 10 == 0:
            logger.info("[Train] Epoch:%4d, BatchIdx:%4d, Cost:%.6f, Acc:%.4f, Speed: %.2f samples/sec" % (epoch, sidx+1, \
                        np.mean(all_cost), count/((sidx+1)*batch_size), (sidx+1)*batch_size/(time.time()-start_time)))
    y_predict = np.concatenate(y_predict)
    y_truth = np.concatenate(y_truth)
    # calculate auc
    fpr, tpr, thresholds = metrics.roc_curve(y_truth, y_predict)
    auc_value = metrics.auc(fpr, tpr)
    # get confusion matrix
    y_predict_int = (y_predict>=0.5).astype("int")
    cm = confusion_matrix(y_truth, y_predict_int)
    return count/((sidx+1)*batch_size), cm, np.mean(all_cost), auc_value

def valid(gvs, model, data_loader, epoch):
    batch_size = gvs.batch_size
    all_cost = []
    count = 0
    start_time = time.time()
    y_predict = []
    y_truth = []
    for sidx, [data, label, image_path] in enumerate(data_loader, 0):
        error, model_output = model.predict(data, label)
        # ----- collect predicts and labels ----- #
        y_predict.append(model_output.cpu().numpy()[:, 1])
        y_truth.append(label.numpy())
        # --------------------------------------- #
        all_cost.append(error)
        count += model.is_correct(label, model_output)
        
        if (sidx+1) % 10 == 0:
            logger.info("[Valid] Epoch:%4d, BatchIdx:%4d, Cost:%.6f, Acc:%.4f, Speed: %.2f samples/sec" % (epoch, sidx+1, \
                        np.mean(all_cost), count/((sidx+1)*batch_size), (sidx+1)*batch_size/(time.time()-start_time)))
    y_predict = np.concatenate(y_predict)
    y_truth = np.concatenate(y_truth)
    # calculate auc
    fpr, tpr, thresholds = metrics.roc_curve(y_truth, y_predict)
    auc_value = metrics.auc(fpr, tpr)
    # get confusion matrix
    y_predict_int = (y_predict>=0.5).astype("int")
    cm = confusion_matrix(y_truth, y_predict_int)
    return count/((sidx+1)*batch_size), cm, np.mean(all_cost), auc_value
    
def load_data():
    train_gvs = GVS()
    test_gvs = GVS()
    test_gvs.batch_size = 12

    train_list, test_list = get_filename_list(train_gvs)
    print("Loaded %d train and %d test samples!" % (len(train_list), len(test_list)))
    # transform
    train_transform = transforms.Compose([transforms.Resize(size=(train_gvs.width, train_gvs.height)), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomVerticalFlip(), 
                                          transforms.RandomResizedCrop(size=train_gvs.width, scale=(0.8, 1.0)), 
                                          transforms.RandomRotation(degrees=10), 
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize(size=(test_gvs.width, test_gvs.height)), transforms.ToTensor()])
    
    train_dataset = MalignantFolder(train_list, train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_gvs.batch_size, shuffle=True, num_workers=8)
    test_dataset = MalignantFolder(test_list, test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_gvs.batch_size, shuffle=False, num_workers=8)
    
    return train_gvs, test_gvs, train_loader, test_loader

class GuoConfusion():
    def __init__(self, mat):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError('Confusion matrix should be a squre matrix.')
        self.mat = mat
        
    def TP(self, idx=-1):
        return self.mat[idx,idx] if idx >= 0 else np.diag(self.mat)
    
    def FP(self, idx=-1):
        return self.mat[:,idx].sum() - self.mat[idx,idx] if idx >= 0 else np.sum(self.mat,0) - np.diag(self.mat)
    
    def FN(self, idx=-1):
        return self.mat[idx,:].sum() - self.mat[idx,idx] if idx >= 0 else np.sum(self.mat,1) - np.diag(self.mat)
    
    def TN(self, idx=-1):
        return self.mat.sum() - self.TP(idx) - self.FP(idx) - self.FN(idx)
        
    def recall(self, idx=-1):
        return self.TP(idx) / (self.TP(idx) + self.FN(idx))
    
    def precision(self, idx=-1):
        return self.TP(idx) / (self.TP(idx) + self.FP(idx))
    
    def f1(self, idx=-1):
        return 2 * self.TP(idx) / (2 * self.TP(idx) + self.FP(idx) + self.FN(idx))
    
    def accuracy(self):
        return np.diag(self.mat).sum() / self.mat.sum()
    
    def everything(self):
        ml = ['recall', 'precision', 'f1', 'accuracy']
        return dict([(func, getattr(self, func)().mean()) for func in ml])

    def everything_str(self):
        ret = ' | '.join(['%s:%.4f'%(k,v) for k, v in self.everything().items()])
        return ret
    
    def recall_str(self):
        ret = " ".join(["%.4f" % self.recall()[i] for i in range(self.mat.shape[0])])
        return ret
    
    def paper_log(self):
        tn = self.mat[0,0]
        fp = self.mat[0,1]
        fn = self.mat[1,0]
        tp = self.mat[1,1]
        return tp, tn, fp, fn
    
if __name__ == "__main__":
    main()