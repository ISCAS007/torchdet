# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as td
import torch
import time
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import argparse
from easydict import EasyDict as edict

class cls_dataset(td.Dataset):
    def __init__(self,config,split='train'):
        super().__init__()
        self.config=config
        self.split=split
        self.img_size=config.img_size
        if config.dataset_name=='github_cair':
            image_dataset=github_cair(config.root_path)
        else:
            assert False
            
        x_train,x_test,y_train,y_test=image_dataset.get_train_val()
        if split=='train':
            self.x=x_train
            self.y=y_train
        elif split in ['val','test']:
            self.x=x_test
            self.y=y_test
        else:
            assert False
        
        self.count=0
        self.size=len(self.x)
        assert self.size>0,print(config)
        assert len(self.x)==len(self.y)
        print('{} dataset {} size={}'.format(config.dataset_name,split,self.size))
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.size
    
    def __next__(self):
        if self.count>=self.size:
            raise StopIteration
        
        path=self.x[self.count]
        origin_img=cv2.imread(path)
        post_img=self.preprocess(origin_img)
        label=self.y[self.count]
        
        self.count+=1
        return {'path':path,
                'post_img':post_img,
                'origin_img':origin_img,
                'label':label}
    
    def __getitem__(self,idx):
        self.count=idx
        
        data=self.__next__()
        new_data={}
        new_data['post_img']=data['post_img']
        new_data['label']=data['label']
        return new_data

    def preprocess(self,pre_img):
        # Padded resize
        img=cv2.resize(pre_img,tuple(self.img_size),interpolation=cv2.INTER_LINEAR)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        return img
        
class github_cair:
    def __init__(self,rootpath):
        self.pos_files=glob.glob(os.path.join(rootpath,'Fire*','*.*'))
        self.neg_files=glob.glob(os.path.join(rootpath,'Normal*','*.*'))
        
        self.pos_files.sort()
        self.neg_files.sort()
        
        self.names=['normal','fire']
    
    def get_train_val(self,test_size=0.33,random_state=25):
        x=self.pos_files+self.neg_files
        y=[1]*len(self.pos_files)+[0]*len(self.neg_files)
        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
        return x_train,x_test,y_train,y_test        
        
class cls_metric():
    def __init__(self):
        self.reset()
    
    def update(self,prediction,label,loss):
        self.tp+=torch.sum(prediction==label)
        self.count+=len(label)
        self.loss+=loss
    
    def get_metric(self):
        assert torch.is_tensor(self.tp)
        assert torch.is_tensor(self.loss)
        assert self.count>0
        return self.tp.item()/self.count,self.loss.item()/self.count
    
    def reset(self):
        self.tp=0
        self.count=0
        self.loss=0
    
class trainer:
    def __init__(self,config):
        self.config=config
        
        ## set dataset
        self.dataset={}
        for split in ['train','val']:
            batch_size=config.batch_size if split=='train' else 1
            shuffle=True if split=='train' else False
            drop_last=True if split=='train' else False
            self.dataset[split]=td.DataLoader(dataset=cls_dataset(config,split),
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
        self.metric=cls_metric()
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir = os.path.join(config.log_dir, config.model_name,
                               config.dataset_name, config.note, time_str)
        
        self.writer=self.init_writer(config,log_dir)
        
    def get_model(self):
        model_urls = {
            'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
            'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
            'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
            'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
            'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        }
        
        from torchvision.models import vgg11,vgg13
        import torch.utils.model_zoo as model_zoo
        
        model=locals()[self.config.model_name](pretrained=False,num_classes=2)
        
        load_state_dict=model_zoo.load_url(model_urls[self.config.model_name])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in load_state_dict.items() if k.find('classifier.6')==-1}
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
        
        acc,loss=self.metric.get_metric()
        self.writer.add_scalar('{}/loss'.format(split),loss,epoch)
        self.writer.add_scalar('{}/acc'.format(split),acc,epoch)
            
    def validation(self,epoch):
        self.train(epoch,split='val')
        
    def train_val(self):
        tqdm_epoch = trange(self.config.epoch, desc='epoch', leave=True)
        for epoch in tqdm_epoch:
            self.train(epoch)
            acc,loss=self.metric.get_metric()
            tqdm_epoch.set_postfix(train_acc=acc,loss=loss)
            
            with torch.no_grad():
                self.validation(epoch)
            acc,loss=self.metric.get_metric()
            tqdm_epoch.set_postfix(val_acc=acc,loss=loss)
                
        self.writer.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        help='model name',
                        choices=['vgg11','vgg13'],
                        default='vgg11')
    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['github_cair'],
                        default='github_cair')
    
    parser.add_argument('--epoch',
                        help='train epoch',
                        type=int,
                        default=30)
    
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=4)
    return parser

def get_default_config():
    config=edict()
    config.model_name='vgg11'
    config.dataset_name='github_cair'
    config.root_path=os.path.join('dataset','smoke',config.dataset_name)
    config.img_size=(224,224)
    config.batch_size=2
    config.epoch=30
    config.lr=1e-4
    config.note='smoke'
    config.log_dir=os.path.expanduser('~/tmp/logs/torchdet')
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
            
    t=trainer(config)
    t.train_val()
            