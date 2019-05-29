# -*- coding: utf-8 -*-
import cv2
import torch.utils.data as td
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np

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

class video2img(github_cair):
    def __init__(self,root_path):
        files=glob.glob(os.path.join(root_path,'**','*.jpg'),recursive=True)
        
        self.pos_files=[]
        self.neg_files=[]
        for f in files:
            basename=os.path.basename(f)
            if basename.find('normal')>=0:
                self.neg_files.append(f)
            else:
                assert basename.find('smoke')>=0 or basename.find('fire')>=0
                self.pos_files.append(f)
        
        self.pos_files.sort()
        self.neg_files.sort()
        
        self.names=['normal','fire_or_smoke']
        
        
class cls_dataset(td.Dataset):
    def __init__(self,config,split='train'):
        super().__init__()
        self.config=config
        self.split=split
        self.img_size=config.img_size
        if config.dataset_name=='github_cair':
            image_dataset=github_cair(config.root_path)
        elif config.dataset_name in ['CVPRLab','FireSense','VisiFire']:
            image_dataset=video2img(config.root_path)
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