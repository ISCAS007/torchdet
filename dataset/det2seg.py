# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
import cv2
from dataset.coco import CocoDataset,get_transform

def get_dataset(config,split):
    if config.dataset_name=='coco2014':
        set_name='train2014' if split=='train' else 'val2014'
        dataset=CocoDataset(config.root_path,set_name=set_name)
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
    def __init__(self,dataset):
        self.dataset=dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
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
            
        sample['overlap_map']=overlap_map
        if self.transform:
            sample = self.transform(sample)
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