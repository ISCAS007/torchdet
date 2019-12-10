# -*- coding: utf-8 -*-
import os
import glob
import shutil
if __name__ == '__main__':
    src_folder='dataset/smoke/lighter/lighter'
    dir_folder='dataset/smoke/lighter/pos'

    img_files=glob.glob(os.path.join(src_folder,'*'))
    assert len(img_files)>0
    print(len(img_files))


    for idx,img in enumerate(img_files):
        root,ext=os.path.splitext(img)
        target_img=os.path.join(dir_folder,str(idx)+ext)
        print('copy {} to {}'.format(img,target_img))
        shutil.copy(img,target_img)