# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import skvideo.io

def split_image(image,target_size,draw_split=False):
    """
    target_size=[th,tw]
    """
    if isinstance(target_size,int):
        target_size=(target_size,target_size)
    
    h,w,c=image.shape
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1
    
    imgs=[]
    draw_img=image
    for i in range(h_num):
        for j in range(w_num):
            if i==h_num-1:
                h_end=h
            else:
                h_end=i*th+2*th
            
            if j==w_num-1:
                w_end=w
            else:
                w_end=j*tw+2*tw
            
            
            if draw_split:
                pt1=(j*tw,i*th)
                pt2=(w_end,h_end)
                draw_img=cv2.rectangle(draw_img,pt1,pt2,color=(255,0,0),thickness=5)
            else:
                img=image[i*th:h_end,j*tw:w_end]
                imgs.append(img)
    
    if draw_split:
        return draw_img
    else:
        return imgs
    
def merge_image(imgs,target_size,origin_size):
    if isinstance(target_size,int):
        target_size=(target_size,target_size)
        
    h,w,c=origin_size
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1
    
    image=np.zeros(origin_size,np.uint8)
    for i in range(h_num):
        for j in range(w_num):
            if i==h_num-1:
                h_end=h
            else:
                h_end=i*th+2*th
            
            if j==w_num-1:
                w_end=w
            else:
                w_end=j*tw+2*tw
            
            split_img=imgs[i*w_num+j]
            shape=(h_end-i*th,w_end-j*tw)
            if split_img.shape[0:2]!=shape:
                new_img=cv2.resize(split_img,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
                image[i*th:h_end,j*tw:w_end]=new_img
            else:
                image[i*th:h_end,j*tw:w_end]=split_img
                
    return image

class yolov3_loadImages:
    def __init__(self,path,img_size=[416,416]):
        if isinstance(img_size,int):
            img_size=(img_size,img_size)
            
        self.img_size=img_size
        files=glob.glob(os.path.join(path,'**','*'),recursive=True)
        suffix=('jpg','png','jpeg','bmp')
        self.img_files=[f for f in files if f.lower().endswith(suffix)]
        self.count=0
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.img_files)
    
    def __next__(self):
        if self.count>=len(self.img_files):
            raise StopIteration
            
        path=self.img_files[self.count]
        origin_img=cv2.imread(path)
        self.count+=1
        split_imgs=split_image(origin_img,self.img_size)
        
        resize_imgs=[self.preprocess(img) for img in split_imgs]
        return path,resize_imgs,split_imgs,origin_img
    
    def preprocess(self,pre_img):
        # Padded resize
        img=cv2.resize(pre_img,self.img_size,interpolation=cv2.INTER_LINEAR)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        return img
    
class yolov3_loadVideos(yolov3_loadImages):
    def __init__(self,path,img_size=[416,416]):
        super().__init__(path,img_size)
        files=glob.glob(os.path.join(path,'**','*'),recursive=True)
        suffix=('mp4','avi','mov','wmv')
        self.video_files=[f for f in files if f.lower().endswith(suffix)]
        self.count=0
        self.cap=None
        self.frame=0
        self.nframes=0
    
    def __len__(self):
        return len(self.video_files)
    
    def __next__(self):
        if self.cap is None:
            self.new_video()
            
        returned=False
        while not returned:
            while True:
                origin_img=self.cap.__next__()
                self.frame+=1
                
                if self.frame%30==0:
                    returned=True
                    break
                elif self.frame>=self.nframes:
                    break
            
            if returned:
                return_img=origin_img
                break
            else:
                print('frame={},total frame={},video_path'.format(self.frame,
                      self.nframes,
                      self.video_files[self.count]))

                self.cap.close()
                if self.count+1>=len(self.video_files):
                    raise StopIteration
                else:
                    self.count+=1
                    self.new_video()
        
        split_imgs=split_image(return_img,self.img_size)
        print(self.frame,self.nframes)
        
        resize_imgs=[self.preprocess(img) for img in split_imgs]
        return self.video_files[self.count],resize_imgs,split_imgs,origin_img
    
    def new_video(self):
        self.frame = 0
        cv_cap = cv2.VideoCapture(self.video_files[self.count])
        self.nframes = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv_cap.release()
        
        self.cap = skvideo.io.vreader(self.video_files[self.count])
    
if __name__ == '__main__':
    files=glob.glob(os.path.join('dataset/demo','**','*'),recursive=True)
    suffix=('jpg','png','jpeg','bmp')
    img_files=[f for f in files if f.lower().endswith(suffix)]
    
    target_size=(224,224)
    for img_f in img_files:
        image=cv2.imread(img_f)
        draw_img=split_image(image,target_size,draw_split=True)
        plt.imshow(draw_img)
        plt.show()
        
        break