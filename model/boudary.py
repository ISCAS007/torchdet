# -*- coding: utf-8 -*-

import os
import argparse
from easydict import EasyDict as edict
from model.overlap import trainer
from dataset.seg2boundary import get_dataset

class boundary_trainer(trainer):
    def __init__(self,config):
        super().__init__(config)
    
    def get_dataset(self):
        """
        get data loader
        """
        self.dataset={}
        for split in ['train','val']:
            self.dataset[split]=get_dataset(self.config,split)
            
    def get_data(self,data):
        """
        get input and annotation from data loader
        """
        inputs,label=data
        return inputs.to(self.device),label.to(self.device).long()
    
def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--app',
                        help='train or vis',
                        choices=['train','vis'],
                        default='train')
    
    parser.add_argument('--model_name',
                        help='model name',
                        choices=['fcn_resnet50','fcn_resnet101'],
                        default='fcn_resnet50')
    
    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['voc2012','cityscapes','SBD'],
                        default='voc2012')
    
    parser.add_argument('--epoch',
                        help='train epoch',
                        type=int,
                        default=30)
    
    parser.add_argument('--batch_size',
                        help='batch size (worked when app=train, be 1 for app=vis)',
                        type=int,
                        default=2)
    
    parser.add_argument('--save_model',
                        help='save the model or not(default False)',
                        action='store_true')
    
    parser.add_argument('--load_model_path',
                        help='model weight path')
    
    return parser

def get_default_config():
    config=edict()
    config.app='train'
    config.model_name='fcn_resnet50'
    config.dataset_name='voc2012'
#    config.img_size=(224,224)
    config.batch_size=2
    config.epoch=30
    config.lr=1e-4
    config.note='boundary'
    config.log_dir=os.path.expanduser('~/tmp/logs/torchdet')
    config.save_model=False
    config.load_model_path=None
    return config

def finetune_config(config):
    if config.dataset_name=='voc2012':
        config.root_path=os.path.expanduser('~/cvdataset/VOC')
    elif config.dataset_name=='cityscapes':
        config.root_path=os.path.expanduser('~/cvdataset/Cityscapes/leftImg8bit_trainvaltest')
    elif config.dataset_name=='SBD':
        config.root_path=os.path.expanduser('~/cvdataset/VOC/benchmark_RELEASE/dataset')
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
    t=boundary_trainer(config)
    if config.app=='train':
        t.train_val()
    else:
        t.visulization()
            