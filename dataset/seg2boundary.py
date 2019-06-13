# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,RandomSampler,DataLoader,SequentialSampler
from skimage.segmentation import find_boundaries
from torchvision.datasets import VOCSegmentation,Cityscapes,SBDataset
from util.segmentation import transforms as T
from util.segmentation.utils import collate_fn
 
def get_transform(split):
    train = split=='train'
    base_size = 520
    crop_size = 480

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

def get_dataset(config,split):
    if config.dataset_name=='voc2012':
        image_set='train' if split=='train' else 'val'
        # image_set in ['train','val','trainval']
        dataset=VOCSegmentation(root=config.root_path,year='2012',image_set=image_set)
    elif config.dataset_name=='cityscapes':
        # train, test or val if mode=”gtFine” otherwise train, train_extra or val
        dataset=Cityscapes(root=config.root_path,split=split,mode='fine',target_type='semantic')
    elif config.dataset_name=='SBD':
        image_set='train' if split=='train' else 'val'
        # Select the image_set to use, train, val or train_noval. Image set train_noval excludes VOC 2012 val images.
        dataset=SBDataset(root=config.root_path,image_set=image_set,mode='segmentation')
    else:
        assert False
    
    transform=get_transform(split)
    
    boundary_dataset = Seg2Boundary(dataset,transform)
    batch_size=config.batch_size if split=='train' else 1
    drop_last=True if split=='train' else False
    if split=='train':
        sampler=RandomSampler(boundary_dataset)
    else:
        sampler=SequentialSampler(boundary_dataset)
    data_loader=DataLoader(dataset=boundary_dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           sampler=sampler,
                           drop_last=drop_last)
    
    return data_loader

def get_boundary_distance(label_img):
    boundary=find_boundaries(label_img,mode='thick').astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    boundary_binary=boundary.copy()
    for i in range(4):
        boundary_binary = cv2.dilate(boundary_binary,kernel,iterations = 1)
        boundary+=boundary_binary

    return boundary

def get_boundary_offset(label_img):
    #todo
    pass

class Seg2Boundary(Dataset):
    """
    convert segmentation dataset to boundary segmentation dataset
    """
    def __init__(self,dataset,transform=None):
        self.dataset=dataset
        self.transform=transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        img,label=self.dataset[idx]
        label=np.array(label)
        boundary=get_boundary_distance(label)
        
        if self.transform:
            img,boundary=self.transform(img,Image.fromarray(boundary))
            
        return img,boundary
    
    
    