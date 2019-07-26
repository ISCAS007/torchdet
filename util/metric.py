# -*- coding: utf-8 -*-
import numpy as np
import torch

class ClsMetric(object):
    def __init__(self):
        self.reset()
    
    def update(self,label_preds,label_trues,loss):
        self.tp+=torch.sum(label_preds==label_trues)
        self.count+=len(label_trues)
        self.loss+=loss
    
    def get_metric(self):
        assert torch.is_tensor(self.tp)
        assert torch.is_tensor(self.loss)
        assert self.count>0
        return {'acc':self.tp.item()/self.count,
                'loss':self.loss.item()/self.count}
    
    def reset(self):
        self.tp=0
        self.loss=0
        self.count=0

class SegMetric(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self,label_preds,label_trues,loss):
        if isinstance(label_trues,torch.Tensor):
            label_trues=label_trues.data.cpu().numpy()
        if isinstance(label_preds,torch.Tensor):
            label_preds=label_preds.data.cpu().numpy()
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        
        self.loss+=loss
        self.count+=1
    
    def get_metric(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
#        acc_cls = np.diag(hist) / hist.sum(axis=1)
        diag=np.diag(hist)
        acc_cls = np.divide(diag,hist.sum(axis=1),out=np.zeros_like(diag),where=diag!=0)
        
        acc_cls = np.nanmean(acc_cls)
        iu = np.divide(diag,(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)),out=np.zeros_like(diag),where=diag!=0)
        mean_iu = np.nanmean(iu)
        #freq = hist.sum(axis=1) / hist.sum()
        #fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        #cls_iu = dict(zip(range(self.n_classes), iu))
        
        if isinstance(self.loss,torch.Tensor):
            loss=self.loss.item()/self.count
        else:
            loss=self.loss/self.count

        return {'acc': acc,
                'acc_cls': acc_cls,
                #'fwavacc': fwavacc,
                'miou': mean_iu,
                'loss':loss}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.loss=0
        self.count=0