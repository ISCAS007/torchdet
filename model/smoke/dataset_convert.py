# -*- coding: utf-8 -*-

"""
convert dataset to a standard format
xxx_train.txt --> xxx training images
xxx_test.txt --> xxx test images

xxx in [fire,smoke,normal]
"""

import os
import glob
import warnings

from easydict import EasyDict as edict


def get_img_files(root_path,img_suffix):
    files=glob.glob(os.path.join(root_path,'**','*.*'),recursive=True)
    img_files=[f for f in files if f.lower().endswith(img_suffix)]

    print(root_path,len(img_files))
    return img_files

def convert_dataset_to_txt(config):
    img_suffix=('jpg','png','jpeg','bmp')
    if config.dataset_name=='BowFire':
        # fire, not_fire, smoke, normal
        # no training set for smoke, and smoke image is very small

        config.root_path='dataset/smoke/BowFire/BoWFireDataset'
        train_csv=os.path.join(config.root_path,'dataset.csv')
        test_csv=os.path.join(config.root_path,'trainset.csv')

        for file_csv,file_txt in zip([train_csv,test_csv],['train.txt','test.txt']):
            fire_file_txt=os.path.join(config.root_path,'fire_'+file_txt)
            normal_file_txt=os.path.join(config.root_path,'normal_'+file_txt)
            smoke_file_txt=os.path.join(config.root_path,'smoke_'+file_txt)

            fire_writer=open(fire_file_txt,'w')
            normal_writer=open(normal_file_txt,'w')
            smoke_writer=open(smoke_file_txt,'w')
            with open(file_csv,'r') as f:
                for l in f.readlines():
                    try:
                        img,gt=l.split(sep=',')
                    except:
                        # remove comment on csv
                        continue

                    img=img.strip()
                    img_path=os.path.join(config.root_path,img)
                    assert os.path.exists(img_path),'img_path {}'.format(img_path)
                    img_path=os.path.relpath(img_path)

                    if img.find('not_fire')==-1 and img.find('fire')!=-1:
                        fire_writer.write(img_path+'\n')
                    elif img.find('not_fire')!=-1 or img.find('normal')!=-1:
                        normal_writer.write(img_path+'\n')
                    elif img.find('smoke')!=-1:
                        smoke_writer.write(img_path+'\n')
                    else:
                        warnings.warn('unknown img {}'.format(img_path))

            fire_writer.close()
            normal_writer.close()
            smoke_writer.close()

    elif config.dataset_name=='dunnings':
        config.root_path='dataset/smoke/dunnings/fire-dataset-dunnings/images-224x224'

        img_files=[]
        for split in ['train','test']:
            img_files+=get_img_files(os.path.join(config.root_path,split),img_suffix)

        writers={}
        for split in ['train','test']:
            for label in ['fire','normal']:
                txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
                writers[split+label]=open(txt_path,'w')

        for file in img_files:
            if file.find('train')!=-1:
                split='train'
            else:
                split='test'

            assert file.find(split)!=-1,file

            if file.find('nofire')!=-1:
                label='normal'
                flag='nofire'
            else:
                label='fire'
                flag='fire'

            assert file.find(flag)!=-1

            writers[split+label].write(file+'\n')

        for split in ['train','test']:
            for label in ['fire','normal']:
                writers[split+label].close()
    elif config.dataset_name=='kaggle_cctv':
        config.root_path='dataset/smoke/kaggle/cctv/data/img_data'
        img_files=get_img_files(config.root_path,img_suffix)

        writers={}
        for split in ['train','test']:
            for label in ['fire','normal','smoke']:
                txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
                writers[split+label]=open(txt_path,'w')

        for file in img_files:
            pathname=os.path.dirname(file)
            if pathname.find('train')!=-1:
                split='train'
            else:
                split='test'

            assert pathname.find(split)!=-1,file

            if pathname.find('default')!=-1:
                label='normal'
                flag='default'
            elif pathname.find('fire')!=-1:
                flag=label='fire'
            else:
                flag=label='smoke'

            assert pathname.find(flag)!=-1

            writers[split+label].write(file+'\n')

        for split in ['train','test']:
            for label in ['fire','normal','smoke']:
                writers[split+label].close()

    elif config.dataset_name=='kaggle_fire_detection':
        config.root_path='dataset/smoke/kaggle/Fire-Detection'
        img_files=get_img_files(config.root_path,img_suffix)
        # fire-detection, not split with train and test


        writers={}
        split='test'
        for label in ['fire','normal']:
            txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
            writers[split+label]=open(txt_path,'w')

        for file in img_files:
            if file.find('Fire-Detection/0')!=-1:
                label='normal'
            elif file.find('Fire-Detection/1')!=-1:
                label='fire'
            else:
                assert False,file

            writers[split+label].write(file+'\n')

        for label in ['fire','normal']:
            writers[split+label].close()
    elif config.dataset_name in ['CVPRLab_img']:
        config.root_path=os.path.join('dataset/smoke',config.dataset_name)
        img_files=get_img_files(config.root_path,img_suffix)

        names=['fire','normal','smoke']
        writers={}
        split='train'
        for label in names:
            txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
            writers[split+label]=open(txt_path,'w')

        for file in img_files:
            base_name=os.path.basename(file)
            for label in names:
                if base_name.find(label)!=-1:
                    writers[split+label].write(file+'\n')

        for label in names:
            writers[split+label].close()
    elif config.dataset_name in ['FireSense_img']:
        config.root_path=os.path.join('dataset/smoke',config.dataset_name)
        img_files=get_img_files(config.root_path,img_suffix)

        names=['fire','normal','smoke']
        writers={}
        split='train'
        for label in names:
            txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
            writers[split+label]=open(txt_path,'w')

        for file in img_files:
            base_name=os.path.relpath(file,start=config.root_path)
            dir_name=os.path.dirname(base_name)
            for label in names:
                if dir_name.find(label)!=-1 and dir_name.find('pos')!=-1:
                    writers[split+label].write(file+'\n')

                if dir_name.find('neg')!=-1:
                    writers[split+'normal'].write(file+'\n')

        for label in names:
            writers[split+label].close()
    elif config.dataset_name in ['github_cair']:
        config.root_path=os.path.join('dataset/smoke',config.dataset_name)
        img_files=get_img_files(config.root_path,img_suffix)

        names=['fire','normal']
        writers={}
        split='train'
        for label in names:
            txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
            writers[split+label]=open(txt_path,'w')

        for file in img_files:
            base_name=os.path.relpath(file,start=config.root_path)
            dir_name=os.path.dirname(base_name).lower()
            for label in names:
                if dir_name.find(label)!=-1:
                    writers[split+label].write(file+'\n')

        for label in names:
            writers[split+label].close()
    elif config.dataset_name in ['VisiFire_img']:
        config.root_path=os.path.join('dataset/smoke',config.dataset_name)
        img_files=get_img_files(config.root_path,img_suffix)

        names=['fire','smoke']
        writers={}
        split='train'
        for label in names:
            txt_path=os.path.join(config.root_path,label+'_'+split+'.txt')
            writers[split+label]=open(txt_path,'w')

        for file in img_files:
            base_name=os.path.relpath(file,start=config.root_path)
            dir_name=os.path.dirname(base_name).lower()
            for label in names:
                if dir_name.find(label)!=-1:
                    writers[split+label].write(file+'\n')

        for label in names:
            writers[split+label].close()
    else:
        assert False

    return config

if __name__ == '__main__':
    config=edict()

    for dataset in ['github_cair','FireSense_img','CVPRLab_img',
                    'kaggle_fire_detection','kaggle_cctv','dunnings','BowFire',
                    'VisiFire_img']:
        config.dataset_name=dataset
        convert_dataset_to_txt(config)