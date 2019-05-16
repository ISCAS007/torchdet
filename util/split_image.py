# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def split_image(image,target_size,draw_split=False):
    """
    target_size=[th,tw]
    """
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