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
import numpy as np
from model.smoke.dataset import cls_dataset,simple_preprocess
import yaml
import glob
import warnings
import cv2

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
        if config.app in ['train']:
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
        
        if config.app in ['train']:
            self.loss_fn=torch.nn.CrossEntropyLoss()
            ## set optimizer
            optimizer_params = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
            self.optimizer=torch.optim.Adam(optimizer_params,lr=config.lr)
            self.metric=cls_metric()
            time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
            self.log_dir = os.path.join(config.log_dir, config.model_name,
                                   config.dataset_name, config.note, time_str)
            
            self.writer=self.init_writer(config,self.log_dir)
        elif config.app in ['test']:
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
        elif config.app in ['vis']:
            pass
        else:
            assert False
            
        
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
        
        if self.config.save_model:
            save_model_path=os.path.join(self.log_dir,'model_{}.pt'.format(self.config.epoch))
            print('save model to',save_model_path)
            torch.save(self.model,save_model_path)
            
    def test(self,image):
        assert isinstance(image,np.ndarray)
        image=simple_preprocess(image,config.img_size)
        inputs=torch.unsqueeze(torch.from_numpy(image),dim=0).to(self.device)
        outputs=self.model.forward(inputs)
        result=torch.softmax(outputs,dim=1).data.cpu().numpy()
        return np.squeeze(result)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app',
                        help='train(train and val) or vis or test(without ground truth)',
                        choices=['train','vis','test'],
                        default='train')
    
    parser.add_argument('--model_name',
                        help='model name',
                        choices=['vgg11','vgg13'],
                        default='vgg11')
    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['github_cair','CVPRLab','FireSense','VisiFire'],
                        default='github_cair')
    
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
    
    parser.add_argument('--load_model_path',
                        help='model dir or model full path')
    
    parser.add_argument('--note',
                        default='smoke')
    return parser

def get_default_config():
    config=edict()
    config.app='train'
    config.model_name='vgg11'
    config.dataset_name='github_cair'
    config.root_path=os.path.join('dataset','smoke',config.dataset_name)
    config.img_size=(224,224)
    config.batch_size=2
    config.epoch=30
    config.lr=1e-4
    config.note='smoke'
    config.log_dir=os.path.expanduser('~/tmp/logs/torchdet')
    config.save_model=False
    return config

def load_config(config_path_or_file,get_default_config):
    """
    config_path_or_file format:
        1. None, load default config
        2. xxx/xxx/xxx.config.txt, load config.txt and merge with default config
        3. xxx/xxx/weight_file, load config.txt and merge with default config
        4. xxx/xxx/ load config.txt and merge with default config
    """
    if config_path_or_file is None:
        return get_default_config()
    elif config_path_or_file.endswith("config.txt"):
        config_file=config_path_or_file
    elif os.path.isdir(config_path_or_file):
        files=glob.glob(os.path.join(config_path_or_file,'**','config.txt'),recursive=True)
        if len(files)==0:
            warnings.warn('no config.txt found on directory {}'.format(config_path_or_file))
            return get_default_config()
        else:
            config_file=files[0]
    elif os.path.isfile(config_path_or_file):
        config_file=os.path.join(os.path.dirname(config_path_or_file),'config.txt')
    else:
        assert False,'unknwon config path format'

    f=open(config_file,'r')
    l=f.readline()
    f.close()

    d=yaml.load(l,Loader=yaml.FullLoader)
    config=edict(d)
    
    default_cfg=get_default_config()
    for key in default_cfg.keys():
        if key not in config.keys():
            config[key]=default_cfg[key]
    
    return config

def finetune_config(config):
    if config.dataset_name=='github_cair':
        config.root_path=os.path.join('dataset','smoke',config.dataset_name)
    elif config.dataset_name in ['CVPRLab','FireSense','VisiFire']:
        config.root_path=os.path.join('dataset','smoke',config.dataset_name+'_img')
    else:
        assert False
    return config
    
if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    
    config=load_config(args.load_model_path,get_default_config)
    if args.app=='train':
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
    else:
        config.app=args.app
        config.load_model_path=args.load_model_path
        
    config=finetune_config(config)
    t=trainer(config)
    if args.app=='train':
        t.train_val()
    elif args.app=='test':
        print('start test'+'*'*30)
        files=glob.glob('dataset/demo/**/*.jpg',recursive=True)
        print(files)
        for f in files:
            img=cv2.imread(f)
            with torch.no_grad():
                result=t.test(img)
            print(f,'%0.2f'%result[1])
            