# -*- coding: utf-8 -*-
"""
bbox-free detection method
prediction object overlap map to generate bbox
"""
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
from dataset.coco import UnNormalizer
from packaging import version
import torchvision 
import numpy as np
import glob
import cv2

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
        if isinstance(label_trues,torch.Tensor):
            label_trues=label_trues.data.cpu().numpy()
        if isinstance(label_preds,torch.Tensor):
            label_preds=label_preds.data.cpu().numpy()
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        
        self.loss+=loss
        self.count+=1
    
    def get_metric(self):
        return self.get_scores()
    
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
        
        if isinstance(self.loss,torch.Tensor):
            loss=self.loss.item()/self.count
        else:
            loss=self.loss/self.count

        return {'acc': acc,
                'acc_cls': acc_cls,
                'fwavacc': fwavacc,
                'miou': mean_iu,
                'loss':loss}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.loss=0
        self.count=0

    
class trainer:
    def __init__(self,config):
        assert version.parse(torchvision.__version__)>=version.parse('0.3.0')
        self.config=config
        """
        for coco, max overlap number is 12
        for PennFudanPed, max overlap number is 3
        """
        self.num_class=config.num_class
        self.get_dataset()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self.get_model()
        self.model.to(self.device)
        self.loss_fn=torch.nn.CrossEntropyLoss()
        self.metric=seg_metric(self.num_class)
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(self.config.log_dir, self.config.model_name,
                               self.config.dataset_name, self.config.note, time_str)
    
    def get_dataset(self):
        self.dataset={}
        for split in ['train','val']:
            self.dataset[split]=get_dataset(self.config,split)
    
    def get_data(self,data):
        inputs=data['img'].to(self.device)
        label=data['overlap_map'].to(self.device).long()
        return inputs,label
            
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
        writer = SummaryWriter(log_dir)
            
        
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
            inputs,label=self.get_data(data)
            label=torch.clamp(label,min=0,max=self.num_class-1)
            outputs=self.model.forward(inputs)['out']
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
        ## set optimizer
        optimizer_params = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
        self.optimizer=torch.optim.Adam(optimizer_params,lr=self.config.lr)
        
        self.writer=self.init_writer(self.config,self.log_dir)

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

    def visulization(self):
        """
        batch size must be 1 for train and val
        """
        model_path=self.config.load_model_path
        if not os.path.exists(model_path):
            print('model path',model_path,'not exist')
            return 0
        
        if os.path.isdir(model_path):
            model_path_list=glob.glob(os.path.join(model_path,'**','*.pt'),recursive=True)
            if len(model_path_list)==0:
                print('cannot find weight file in directory',model_path)
                return 0
                
            print(model_path_list)
            model_path=model_path_list[0]
            print('auto select model',model_path)

        self.model.load_state_dict(torch.load(model_path).state_dict())
        self.model.eval()

        img_unnorm=UnNormalizer()
        with torch.no_grad():
            for split in ['train','val']:
                for i,(data) in enumerate(self.dataset[split]):
                    inputs,label=self.get_data(data)
                    clamp_label=torch.clamp(label,min=0,max=self.num_class-1)
                    outputs=self.model.forward(inputs)['out']
                    loss=self.loss_fn(outputs,clamp_label)
                    predictions=torch.argmax(outputs,dim=1)
                    self.metric.reset()
                    self.metric.update(predictions,clamp_label,loss)
                    metric_dict=self.metric.get_metric()
                    
                    filename=os.path.join('output',split,str(i))
                    img_tensor=img_unnorm.__call__(inputs)
                    save_img=np.transpose(np.squeeze(img_tensor.data.cpu().numpy()),(1,2,0))
                    save_img=(save_img*225).astype(np.uint8)
                    save_label=np.squeeze(predictions.data.cpu().numpy())
                    save_label=(save_label*50).astype(np.uint8)
                    os.makedirs(os.path.dirname(filename),exist_ok=True)
                    cv2.imwrite(filename+'.jpg',save_img)
                    cv2.imwrite(filename+'.png',save_label)
                    print('step {}: miou={},acc={},loss={}, save to {}'.format(i,metric_dict['miou'],
                        metric_dict['acc'],metric_dict['loss'],filename))
                    print('predict label',np.unique(save_label))

                    if i > 10:
                        break

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app',
                        help='train/vis',
                        choices=['train','vis'],
                        default='train')

    parser.add_argument('--model_name',
                        help='model name',
                        choices=['fcn_resnet50','fcn_resnet101'],
                        default='fcn_resnet50')

    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['coco2014','PennFudanPed'],
                        default='coco2014')
    
    parser.add_argument('--epoch',
                        help='train epoch',
                        type=int,
                        default=30)
    
    parser.add_argument('--batch_size',
                        help='batch size (worked when app=train, be 1 for app=vis)',
                        type=int,
                        default=4)
    
    parser.add_argument('--save_model',
                        help='save the model or not',
                        action='store_true')

    parser.add_argument('--load_model_path',
                        help='model weight path to visulization')
    return parser

def get_default_config():
    config=edict()
    config.app='train'
    config.model_name='fcn_resnet50'
    config.dataset_name='coco2014'
    config.img_size=(224,224)
    config.batch_size=2
    config.epoch=30
    config.lr=1e-4
    config.note='overlap'
    config.log_dir=os.path.expanduser('~/tmp/logs/torchdet')
    config.save_model=False
    config.load_model_path=None
    config.num_class=5
    return config

def finetune_config(config):
    if config.dataset_name=='coco2014':
        config.root_path=os.path.expanduser('~/cvdataset/coco')
    elif config.dataset_name=='PennFudanPed':
        config.root_path=os.path.expanduser('~/cvdataset/PennFudanPed')
    else:
        assert False
    
    if config.app=='vis':
        config.batch_size=1
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
    
    if config.app=='train':
        t.train_val()
    else:
        t.visulization()
            