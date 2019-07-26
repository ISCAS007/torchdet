# -*- coding: utf-8 -*-

import torch.utils.data as td
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def get_dataset(config,split,crop_size=224,base_size=256):
    assert split in ['train','val']
    target_path=os.path.join(config.root_path,split)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if split=='train':
        dataset=ImageFolder(target_path,transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        dataset=ImageFolder(target_path, transforms.Compose([
            transforms.Resize(base_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]))
    
    return dataset

def get_loader(config,get_dataset_fn,split,crop_size=224,base_size=256):
    dataset=get_dataset_fn(config,split,crop_size,base_size)
    batch_size=config.batch_size if split=='train' else 1
    shuffle=True if split=='train' else False
    drop_last=True if split=='train' else False
    loader=td.DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)
    return loader
        
        