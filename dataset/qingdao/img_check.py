# -*- coding: utf-8 -*-

import glob
import cv2
import os

# mogrify -set comment 'Extraneous bytes removed' *.jpg
root_dir=os.path.expanduser('~/cvdataset/QingDao/digger_cls3_0713')
files=glob.glob(os.path.join(root_dir,'**','*.*'),recursive=True)
#suffix=('jpg','jpeg','bmp','png')
suffix=('jpg')
img_files=[f for f in files if f.lower().endswith(suffix)]

bad_img=['000025.jpg','000153.jpg','000356.jpg','000867.jpg','001009.jpg']
print(len(img_files))
for f in img_files:
    try:
        if os.path.basename(f) in bad_img:
            print('bad image',f)
        img=cv2.imread(f)
        if os.path.basename(f) in bad_img:
            print('end it')
    except:
        print(f)