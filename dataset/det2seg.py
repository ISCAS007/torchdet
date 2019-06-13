# -*- coding: utf-8 -*-
"""
convert detection dataset to overlap map segmentation dataset
"""
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
from dataset.coco import CocoDataset,get_transform
from dataset.PennFudanPed import PennFudanDataset
from dataset.coco import AspectRatioBasedSampler,collater
from tqdm import trange
from easydict import EasyDict as edict
import os

def get_dataset(config,split):
    if config.dataset_name=='coco2014':
        set_name='train2014' if split=='train' else 'val2014'
        base_dataset=CocoDataset(config.root_path,set_name=set_name)
    elif config.dataset_name=='PennFudanPed':
        base_dataset=PennFudanDataset(config.root_path,split=split)
    else:
        assert False
    
    transform=get_transform(split)
    dataset=Det2Seg(base_dataset,transform)

    batch_size=config.batch_size if split=='train' else 1
    drop_last=True if split=='train' else False
    sampler=AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=drop_last)
    data_loader=DataLoader(dataset=dataset,
                collate_fn=collater,
                batch_sampler=sampler)
    
    return data_loader
    
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

class Det2Seg(Dataset):
    """
    convert detection dataset to segmentation format for bbox-free model
    """
    def __init__(self,dataset,transform=None):
        self.dataset=dataset
        self.transform=transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        assert isinstance(idx,int),'{}:{}'.format(type(idx),idx)
        #sample = {'img': img, 'annot': annot}
        sample=self.dataset.__getitem__(idx)
        h,w,c=sample['img'].shape
        overlap_map=np.zeros((h,w),dtype=np.uint8)
        
        N,M=sample['annot'].shape
        assert M==5
        for idx in range(N):
            x1,y1,x2,y2,label_id=sample['annot'][idx,:]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            if x2>x1 or y2>y1:
                overlap_map[y1:y2,x1:x2]+=1
        
        if self.transform:
            sample = self.transform(sample)
            pad_h,pad_w,c=sample['img'].shape
        
            overlap_map=cv2.resize(overlap_map,dsize=(0,0),
                                   fx=sample['scale'],
                                   fy=sample['scale'],
                                   interpolation=cv2.INTER_NEAREST)
            
            h,w=overlap_map.shape
            pad_map=np.zeros((pad_h,pad_w),dtype=np.uint8)
            pad_map[0:h,0:w]=overlap_map
            overlap_map=pad_map
    
        #overlap_map=np.expand_dims(overlap_map,axis=2)
        sample['overlap_map']=torch.from_numpy(overlap_map)
        return sample
    
    def vis(self,idx):
        sample=self.__getitem__(idx)
        N,M=sample['annot'].shape
        assert M==5
        img=sample['img']
        for idx in range(N):
            x1,y1,x2,y2,label_id=sample['annot'][idx,:]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            if x2>x1 or y2>y1:
                label_name = self.dataset.labels[label_id]
                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                
        return img,sample['overlap_map']
    
    def image_aspect_ratio(self, idx):
        return self.dataset.image_aspect_ratio(idx)

if __name__=='__main__':
    config=edict()
    config.dataset_name='PennFudanPed'
    config.root_path=os.path.expanduser('~/cvdataset/PennFudanPed')
    
    for split in ['train','val']:
        count=[0 for i in range(20)]
        d=get_dataset(config,split)
        for i in trange(len(d)):
            data=d.__getitem__(i)
            max_idx=np.max(data['overlap_map'].data.cpu().numpy())
            if max_idx<20:
                count[max_idx]+=1
            else:
                count[-1]+=1
        print(count)