import copy
import numpy as np

def find_all(origin_str, sub_str):
    indices = [-1]
    last_str = copy.deepcopy(origin_str)
    while last_str.find(sub_str) != -1:
        idx = last_str.find(sub_str)
        last_str = last_str[idx+1:]
        real_idx = indices[-1] + 1 + idx
        indices.append(real_idx)
    return indices[1:]

def readlines(filename, read_type="r", encoding="utf-8"):
    tmp_fp = open(filename, read_type, encoding=encoding)
    lines = tmp_fp.readlines()
    tmp_fp.close()

    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
        lines[i] = lines[i].strip("\r")
    return lines

class MyConfusionMatrix():
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
        return "TP:%d, TN:%d, FP:%d, FN:%d" % (tp, tn, fp, fn)