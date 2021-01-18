import torch
import torch.nn as nn
from global_variables import GVS
from seal_datasets import get_filename_list, MalignantFolder
from seal_utils import MyConfusionMatrix
import torchvision.transforms as transforms
from net_model import NetModel
import datetime, time, os
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn import metrics
import pickle

# -------- Setting logger -------- #
# Time zone is set to Asia/BeiJing(which is UTC-8)
# logger.info() outputs logs to both console and file
# logger.debug() outputs logs only to console
tmp_gvs = GVS()
log_filename_prefix = "Evaluate_" + tmp_gvs.exp_target + "_Test" + str(tmp_gvs.test_index) + "_" + tmp_gvs.net.replace(" ", "_") + "_"
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
    train_gvs, valid_gvs, train_loader, test_loader = load_data()
    model = NetModel(valid_gvs)
    model.load_params(epoch=149)
    
    max_valid_acc = -1.0
    max_valid_auc = -1.0
    max_valid_epoch = -1
    train_acc, train_loss, valid_acc, valid_loss = 0.0, 0.0, 0.0, 0.0
    train_cm, valid_cm = np.zeros((train_gvs.num_classes, train_gvs.num_classes)), np.zeros((valid_gvs.num_classes, valid_gvs.num_classes))
    for epoch in range(149, 150):
        logger.info("Epoch:%4d" % (epoch))
#         train_acc, train_cm, train_loss = valid(train_gvs, model, train_loader, epoch+1)
        valid_acc, valid_cm, valid_loss, valid_auc = valid(valid_gvs, model, test_loader, epoch)
        mc = MyConfusionMatrix(train_cm)
        md = MyConfusionMatrix(valid_cm)
        
        if valid_auc > max_valid_auc:
            max_valid_auc = valid_auc
            max_valid_epoch = epoch + 1
        
        logger.info("[Valid] Epoch:%4d, Train AUC: %.4f" % (epoch, train_acc))
        logger.info("[Valid] Epoch:%4d, Valid AUC: %.4f" % (epoch, valid_auc))
        logger.info("[Valid] Epoch:%4d, Max valid AUC is: %.4f at Epoch %d" % (epoch, max_valid_auc, max_valid_epoch))
        logger.info("[Valid] Epoch:%4d, Train CM: %s, Loss:%.6f" % (epoch+1, mc.paper_log(), train_loss))
        logger.info("[Valid] Epoch:%4d, Valid CM: %s, Loss:%.6f" % (epoch+1, md.paper_log(), valid_loss))
    
def train(gvs, model, data_loader, epoch):
    batch_size = gvs.batch_size
    all_cost = []
    count = 0
    start_time = time.time()
    y_predict = []
    y_truth = []
    for sidx, [data, label] in enumerate(data_loader, 0):
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
    y_predict = (y_predict>=0.5).astype("int")
    y_truth = np.concatenate(y_truth)
    cm = confusion_matrix(y_truth, y_predict)
    return count/((sidx+1)*batch_size), cm, np.mean(all_cost)

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
    y_predict = (y_predict>=0.5).astype("int")
    cm = confusion_matrix(y_truth, y_predict)
    return count/((sidx+1)*batch_size), cm, np.mean(all_cost), auc_value
    
def load_data():
    train_gvs = GVS()
    test_gvs = GVS()
    train_gvs.batch_size = 1
    test_gvs.batch_size = 1

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
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_gvs.batch_size, num_workers=0)
    test_dataset = MalignantFolder(test_list, test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_gvs.batch_size, num_workers=0)
    
    return train_gvs, test_gvs, train_loader, test_loader

if __name__ == "__main__":
    main()