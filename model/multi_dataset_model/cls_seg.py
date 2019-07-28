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
from dataset.places365_standard import get_loader,get_dataset
from torch.utils.data import Dataset,RandomSampler,DataLoader,SequentialSampler
from torchvision.datasets import VOCSegmentation,Cityscapes,SBDataset
from util.segmentation import transforms as T
from util.segmentation.utils import collate_fn
from dataset.utils.warp_loader import MyLoader
from util.metric import ClsMetric,SegMetric
import yaml
import glob
import warnings
import cv2
from torch import nn

def get_transform(split,crop_size = 480, base_size = 520):
    train = split=='train'
    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

class TransformDataset(Dataset):
    def __init__(self,dataset,transforms):
        self.dataset=dataset
        self.transforms=transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        img,label=self.dataset[idx]

        if self.transforms:
            img,label=self.transforms(img,label)

        return img,label

def get_seg_dataloader(config,split):
    transform=get_transform(split)

    if config.dataset_name=='voc2012':
        image_set='train' if split=='train' else 'val'
        # image_set in ['train','val','trainval']
        dataset=VOCSegmentation(root=config.root_path,year='2012',image_set=image_set,transforms=transform)
    elif config.dataset_name=='cityscapes':
        # train, test or val if mode=”gtFine” otherwise train, train_extra or val
        dataset=Cityscapes(root=config.root_path,split=split,mode='fine',target_type='semantic')
        dataset=TransformDataset(dataset,transforms=transform)
    elif config.dataset_name=='SBD':
        image_set='train' if split=='train' else 'val'
        # Select the image_set to use, train, val or train_noval. Image set train_noval excludes VOC 2012 val images.
        dataset=SBDataset(root=config.root_path,image_set=image_set,mode='segmentation',transforms=transform)
    else:
        assert False

#    batch_size=config.batch_size if split=='train' else 1
    batch_size=config.batch_size
    drop_last=True if split=='train' else False
    if split=='train':
        sampler=RandomSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)
    data_loader=DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           sampler=sampler,
                           drop_last=drop_last)

    return data_loader

def get_cls_dataloader(config,split):
    return get_loader(config,get_dataset,split,crop_size=480,base_size=520,val_batch_size=config.batch_size)

class SharedClsModel(nn.Module):
    def __init__(self,config,backbone,num_class):
        super().__init__()
        self.config=config
        self.backbone=backbone
        self.layer2_pool=None
        self.layer3_pool=nn.AvgPool2d(2,2)
        self.layer4_pool=nn.AvgPool2d(2,2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*4,num_class)

    def forward(self,x):
        x=self.backbone.conv1(x)
        x=self.backbone.bn1(x)
        x=self.backbone.relu(x)
        x=self.backbone.maxpool(x)

        x=self.backbone.layer1(x)
        x=self.backbone.layer2(x)
        if self.layer2_pool:
            x=self.layer2_pool(x)

        x=self.backbone.layer3(x)
        if self.layer3_pool:
            x=self.layer3_pool(x)

        x=self.backbone.layer4(x)
        if self.layer4_pool:
            x=self.layer4_pool(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return x

class MultiModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.models=self.get_model()

    def get_model(self):
        self.seg_model=self.get_seg_model()
        self.cls_model=SharedClsModel(self.config,self.seg_model.backbone,365)

        return {'seg':self.seg_model,'cls':self.cls_model}

    def forward(self,input_dict):
        """
        if input_dict contain only part of model input
        just output part of model result
        """
        output_dict={}
        for key,value in input_dict.items():
            if key=='seg':
                output_dict[key]=self.models[key].forward(input_dict[key])['out']
            else:
                output_dict[key]=self.models[key].forward(input_dict[key])

        return output_dict

    def get_seg_model(self):
        model_urls = {
            'fcn_resnet50_coco': None,
            'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
            'deeplabv3_resnet50_coco': None,
            'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
        }

        from torchvision.models.segmentation import fcn_resnet50,fcn_resnet101
        import torch.utils.model_zoo as model_zoo

        model=locals()[self.config.model_name](pretrained=False,num_classes=self.config.num_class)

        model_url=model_urls[self.config.model_name+'_coco']
        if model_url:
            load_state_dict=model_zoo.load_url()
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in load_state_dict.items() if k.find('classifier.4')==-1}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        return model

class MultiMetric:
    def __init__(self,num_classes):
        self.metrics={}
        self.metrics['cls']=ClsMetric()
        self.metrics['seg']=SegMetric(num_classes)

    def reset(self):
        for key in self.metrics.keys():
            self.metrics[key].reset()

    def update(self,outputs,labels,losses):
        for key in outputs.keys():
            predict=torch.argmax(outputs[key],dim=1)
            self.metrics[key].update(predict,labels[key],losses[key])

    def get_metric(self):
        metric={}
        for key in self.metrics.keys():
            result=self.metrics[key].get_metric()
            for sub_key,value in result.items():
                metric[key+'/'+sub_key]=value

        return metric

class MultiTrainer:
    def __init__(self,config):
        self.config=config
        self.model=MultiModel(config)
        self.loss_fn={}
        self.loss_fn['cls']=torch.nn.CrossEntropyLoss()
        self.loss_fn['seg']=torch.nn.CrossEntropyLoss(ignore_index=255)
        optimizer_params = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
        self.optimizer=torch.optim.Adam(optimizer_params,lr=config.lr)
        self.metric=MultiMetric(self.config.num_class)

        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(config.log_dir, config.model_name,
                                   config.dataset_name, config.note, time_str)
        self.writer=self.init_writer(config,self.log_dir)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

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

    def get_dataloader(self,split,key=None):
        """
        1. split=train, key=None
            return merged dataloader for train
        2. split=val, key=cls/seg
            return single dataloader for val
        """
        if split =='train':
            cls_config=self.config.copy()
            cls_config=edict(cls_config)
            cls_config.root_path=os.path.join('~/cvdataset/places365_standard')
            cls_dataloader=get_cls_dataloader(cls_config,split)
            seg_dataloader=get_seg_dataloader(self.config,split)
            return MyLoader([cls_dataloader,seg_dataloader])
        elif key == 'cls':
            cls_config=self.config.copy()
            cls_config=edict(cls_config)
            cls_config.root_path=os.path.join('~/cvdataset/places365_standard')
            return get_cls_dataloader(cls_config,split)
        elif key == 'seg':
            return get_seg_dataloader(self.config,split)
        else:
            assert False

    def get_data(self,data,split,key=None):
        """
        1. split=train, key=None
            return merged dataloader for train
        2. split=val, key=cls/seg
            return single dataloader for val
        """
        inputs={}
        labels={}
        if split=='train':
            (cls_img,cls_label),(seg_img,seg_label)=data
#            print(cls_img.shape,cls_label.shape,seg_img.shape,seg_label.shape)
            inputs['cls']=cls_img.to(self.device)
            labels['cls']=cls_label.to(self.device)
            inputs['seg']=seg_img.to(self.device)
            labels['seg']=seg_label.to(self.device).long()
        elif key=='cls':
            cls_img,cls_label=data
            inputs['cls']=cls_img.to(self.device)
            labels['cls']=cls_label.to(self.device).long()
        elif key=='seg':
            seg_img,seg_label=data
            inputs['seg']=seg_img.to(self.device)
            labels['seg']=seg_label.to(self.device).long()
        else:
            assert False

        if 'seg' in labels.keys():
            label=labels['seg']
            foreground_class_ids = [
            7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

            reIndexed_label=torch.zeros_like(label)+255
            for idx,id in enumerate(foreground_class_ids):
                reIndexed_label=torch.where(label==id,torch.tensor(idx),reIndexed_label)
            labels['seg']=reIndexed_label.to(self.device)
        return inputs,labels

    def get_loss(self,outputs,labels):
        loss={}
        for key,value in outputs.items():
            loss[key]=self.loss_fn[key](value,labels[key])
        return loss

    def write_metric(self,metric_dict,split,epoch):
        for key,value in metric_dict.items():
            if torch.is_tensor(value):
                self.writer.add_scalar('{}/{}'.format(split,key),value.item(),epoch)
            else:
                self.writer.add_scalar('{}/{}'.format(split,key),value,epoch)

    def train(self,epoch):
        self.model.train()
        dataloader=self.get_dataloader('train')
        self.metric.reset()

        tqdm_step = tqdm(dataloader, desc='step', leave=False)
        assert len(dataloader)>0
        for i,(data) in enumerate(tqdm_step):

            inputs,labels=self.get_data(data,'train')
            outputs=self.model.forward(inputs)
            losses=self.get_loss(outputs,labels)

            total_loss=0
            for key,loss in losses.items():
                loss.backward()
                total_loss+=loss
            self.optimizer.step()
            self.optimizer.zero_grad()

            tqdm_step.set_postfix(total_loss=total_loss.item())
            self.metric.update(outputs,labels,losses)

        metric_dict=self.metric.get_metric()
        self.write_metric(metric_dict,'train',epoch)

    def validation(self,epoch):
        self.model.eval()
        self.metric.reset()
        for key in ['cls','seg']:
            dataloader=self.get_dataloader('val',key)
            tqdm_step = tqdm(dataloader, desc='step', leave=False)
            assert len(dataloader)>0
            for i,(data) in enumerate(tqdm_step):
                inputs,labels=self.get_data(data,'val',key)
                outputs=self.model.forward(inputs)
                losses=self.get_loss(outputs,labels)

                total_loss=0
                for key,loss in losses.items():
                    total_loss+=loss

                tqdm_step.set_postfix(total_loss=total_loss.item())
                self.metric.update(outputs,labels,losses)
        metric_dict=self.metric.get_metric()
        self.write_metric(metric_dict,'val',epoch)

    def train_val(self):
        tqdm_epoch = trange(self.config.epoch, desc='epoch', leave=True)
        for epoch in tqdm_epoch:
            self.train(epoch)
            metric_dict=self.metric.get_metric()
            tqdm_epoch.set_postfix(train_miou=metric_dict['seg/miou'],
                                   train_acc=metric_dict['cls/acc'])

            with torch.no_grad():
                self.validation(epoch)
            metric_dict=self.metric.get_metric()
            tqdm_epoch.set_postfix(val_miou=metric_dict['seg/miou'],
                                   val_acc=metric_dict['cls/acc'])

        self.writer.close()

        if self.config.save_model:
            save_model_path=os.path.join(self.log_dir,'model_{}.pt'.format(self.config.epoch))
            print('save model to',save_model_path)
            torch.save(self.model,save_model_path)

class Config:
    def __init__(self):
        pass

    @staticmethod
    def get_config():
        parser=Config.get_parser()
        args = parser.parse_args()

        config=Config.load_config(args.load_model_path,Config.get_default_config)
        # load old model, overwrite config.app with args.app
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

        config=Config.finetune_config(config)
        return config

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--app',
                            help='train(train and val) or vis or test(without ground truth)',
                            choices=['train','vis','test'],
                            default='train')

        parser.add_argument('--model_name',
                            help='model name',
                            choices=['fcn_resnet50','fcn_resnet101'],
                            default='fcn_resnet50')

        parser.add_argument('--dataset_name',
                            help='dataset name',
                            choices=['voc2012','cityscapes','SBD'],
                            default='cityscapes')

        parser.add_argument('--epoch',
                            help='train epoch',
                            type=int,
                            default=30)

        parser.add_argument('--batch_size',
                            help='batch size',
                            type=int,
                            default=4)

        parser.add_argument('--load_model_path',
                        help='model dir or model full path')

        parser.add_argument('--note',
                            default='cls_seg')
        return parser

    @staticmethod
    def get_default_config():
        config=edict()
        config.app='train'
        config.model_name='fcn_resnet50'
        config.dataset_name='cityscapes'
        config.img_size=(224,224)
        config.batch_size=2
        config.epoch=30
        config.lr=1e-4
        config.note='cls_seg'
        config.log_dir=os.path.expanduser('~/tmp/logs/torchdet')
        config.load_model_path=None
        config.save_model=False
        return config

    @staticmethod
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

    @staticmethod
    def finetune_config(config):
        if config.dataset_name=='voc2012':
            config.root_path=os.path.expanduser('~/cvdataset/VOC')
            config.num_class=20
        elif config.dataset_name=='cityscapes':
            config.root_path=os.path.expanduser('~/cvdataset/Cityscapes/leftImg8bit_trainvaltest')
            config.num_class=19
        elif config.dataset_name=='SBD':
            config.root_path=os.path.expanduser('~/cvdataset/VOC/benchmark_RELEASE/dataset')
            config.num_class=2
        else:
            assert False

        return config

if __name__ == '__main__':
    config=Config.get_config()
    model=MultiTrainer(config)
    if config.app=='train':
        model.train_val()
