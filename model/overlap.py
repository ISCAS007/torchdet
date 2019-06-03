# -*- coding: utf-8 -*-
import os
import torch.utils.data as td
import torch
import time
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import argparse
from easydict import EasyDict as edict
from dataset.det2seg import get_dataset
from packaging import version
import torchvision 
import numpy as np

class seg_metric(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds,loss):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        
        self.loss+=loss
        self.count+=1

    def get_scores(self):
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
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        #cls_iu = dict(zip(range(self.n_classes), iu))
        
        return {'acc': acc,
                'acc_cls': acc_cls,
                'fwavacc': fwavacc,
                'miou': mean_iu,
                'loss':self.loss.item()/self.count}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.loss=0
        self.count=0

    
class trainer:
    def __init__(self,config):
        assert version.parse(torchvision.__version__)>=version.parse('0.3.0')
        self.config=config
        self.num_class=5
        ## set dataset
        self.dataset={}
        for split in ['train','val']:
            batch_size=config.batch_size if split=='train' else 1
            shuffle=True if split=='train' else False
            drop_last=True if split=='train' else False
            self.dataset[split]=td.DataLoader(dataset=get_dataset(config,split),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self.get_model()
        self.model.to(self.device)
        self.loss_fn=torch.nn.CrossEntropyLoss()
        ## set optimizer
        optimizer_params = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
        self.optimizer=torch.optim.Adam(optimizer_params,lr=config.lr)
        self.metric=seg_metric(self.num_class)
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(config.log_dir, config.model_name,
                               config.dataset_name, config.note, time_str)
        
        self.writer=self.init_writer(config,self.log_dir)
        
    def get_model(self):
        model_urls = {
            'fcn_resnet50_coco': None,
            'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
            'deeplabv3_resnet50_coco': None,
            'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
        }

        
        from torchvision.models.segmentation import fcn_resnet50,fcn_resnet101
        import torch.utils.model_zoo as model_zoo
        
        model=locals()[self.config.model_name](pretrained=False,num_classes=self.num_class)
        
        model_url=model_urls[self.config.model_name+'_coco']
        if model_url:
            load_state_dict=model_zoo.load_url()
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in load_state_dict.items() if k.find('classifier.4')==-1}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        return model
        
    def init_writer(self,config, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        config_str = json.dumps(config, indent=2, sort_keys=True).replace(
            '\n', '\n\n').replace('  ', '\t')
        writer.add_text(tag='config', text_string=config_str)
    
        # write config to config.txt
        config_path = os.path.join(log_dir, 'config.txt')
        config_file = open(config_path, 'w')
        json.dump(config, config_file, sort_keys=True)
        config_file.close()
    
        return writer
 
    def train(self,epoch,split='train'):
        if split=='train':
            self.model.train()
        else:
            self.model.eval()
        
        self.metric.reset()
        
        tqdm_step = tqdm(self.dataset[split], desc='step', leave=False)
        assert len(self.dataset[split])>0
        for i,(data) in enumerate(tqdm_step):
            if split=='train':
                self.optimizer.zero_grad()
            inputs=data['post_img'].to(self.device)
            label=data['label'].to(self.device)
            outputs=self.model.forward(inputs)
            
            loss=self.loss_fn(outputs,label)
            
            if split=='train':
                loss.backward()
                self.optimizer.step()
            
            tqdm_step.set_postfix(loss=loss.item())
            
            predictions=torch.argmax(outputs,dim=1)
            self.metric.update(predictions,label,loss)
        
        metric_dict=self.metric.get_metric()
        self.writer.add_scalar('{}/loss'.format(split),metric_dict['loss'],epoch)
        self.writer.add_scalar('{}/miou'.format(split),metric_dict['miou'],epoch)
        self.writer.add_scalar('{}/acc'.format(split),metric_dict['acc'],epoch)
            
    def validation(self,epoch):
        self.train(epoch,split='val')
        
    def train_val(self):
        tqdm_epoch = trange(self.config.epoch, desc='epoch', leave=True)
        for epoch in tqdm_epoch:
            self.train(epoch)
            metric_dict=self.metric.get_metric()
            tqdm_epoch.set_postfix(train_miou=metric_dict['miou'],loss=metric_dict['loss'])
            
            with torch.no_grad():
                self.validation(epoch)
            metric_dict=self.metric.get_metric()
            tqdm_epoch.set_postfix(val_miou=metric_dict['miou'],loss=metric_dict['loss'])
                
        self.writer.close()
        
        if self.config.save_model:
            save_model_path=os.path.join(self.log_dir,'model_{}.pt'.format(self.config.epoch))
            print('save model to',save_model_path)
            torch.save(self.model,save_model_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        help='model name',
                        choices=['fcn_resnet50','fcn_resnet101'],
                        default='fcn_resnet50')
    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['coco2014'],
                        default='coco2014')
    
    parser.add_argument('--epoch',
                        help='train epoch',
                        type=int,
                        default=30)
    
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=4)
    
    parser.add_argument('--save_model',
                        help='save the model or not',
                        action='store_true')
    return parser

def get_default_config():
    config=edict()
    config.model_name='fcn_resnet50'
    config.dataset_name='coco2014'
    config.root_path=os.path.join('dataset','coco')
    config.img_size=(224,224)
    config.batch_size=2
    config.epoch=30
    config.lr=1e-4
    config.note='smoke'
    config.log_dir=os.path.expanduser('~/tmp/logs/torchdet')
    config.save_model=False
    return config

def finetune_config(config):
    if config.dataset_name=='coco2014':
        config.root_path=os.path.join('dataset','coco')
    else:
        assert False
    return config
    
if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    
    config=get_default_config()
    sort_keys=sorted(list(config.keys()))
    for key in sort_keys:
        if hasattr(args,key):
            print('{} = {} (default: {})'.format(key,args.__dict__[key],config[key]))
            config[key]=args.__dict__[key]
        else:
            print('{} : (default:{})'.format(key,config[key]))
    
    for key in args.__dict__.keys():
        if key not in config.keys():
            print('{} : unused keys {}'.format(key,args.__dict__[key]))
    
    config=finetune_config(config)
    t=trainer(config)
    t.train_val()
            