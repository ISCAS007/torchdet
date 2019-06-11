# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from dataset.coco import CocoDataset,get_transform
from dataset.PennFudanPed import PennFudanDataset

def get_dataset(config,split):
    if config.dataset_name=='coco2014':
        set_name='train2014' if split=='train' else 'val2014'
        dataset=CocoDataset(config.root_path,set_name=set_name)
    elif config.dataset_name=='PennFudanPed':
        dataset=PennFudanDataset(config.root_path,split=split)
    else:
        assert False
    
    transform=get_transform(split)
    return Det2Seg(dataset,transform)
    
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
            overlap_map=cv2.resize(overlap_map,dsize=(0,0),
                                   fx=sample['scale'],
                                   fy=sample['scale'],
                                   interpolation=cv2.INTER_NEAREST)
            pad_h,pad_w,c=sample['img'].shape
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