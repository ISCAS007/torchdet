# -*- coding: utf-8 -*-
"""
convert QingDao digger dataset to darknet format
- https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
remove bad image 
- mogrify -set comment 'Extraneous bytes removed' *.jpg
"""
import sys
import os
import glob
import shutil
import cv2
import argparse
from datetime import datetime
import xml.etree.ElementTree as ET
from jinja2 import Environment, FileSystemLoader
from easydict import EasyDict as edict
import random
import warnings
from tqdm import trange

def check_img(img_f):
    try:
        img=cv2.imread(img_f)
    except:
        return False
    else:
        if img is None:
            return False
        else:
            return True
        
def rename_tag(tag):
    if tag.lower() in ['excavating','excavator']:
        return 'excavator'
    elif tag.lower() in ['b-truck','s-truck','truck']:
        return 'truck'
    elif tag.lower() in ['loader']:
        return 'loader'
    else:
        return 'unknown'
    
def get_files(raw_dir):
    child_files=[]
    for d in os.listdir(raw_dir):
        child_dir=os.path.join(raw_dir,d)
        if os.path.isdir(child_dir):
            print('parse dir',child_dir)
            child_files+=glob.glob(os.path.join(child_dir,'**','*.*'),recursive=True)
    root_files=glob.glob(os.path.join(raw_dir,'*.*'),recursive=False)
    
    print('child files',len(child_files))
    print('root files',len(root_files))
    return root_files+child_files
    
class darknet_pipeline():
    """
    ## pipeline 1
    xxx.names --> class_num --> xxx.cfg
    self.save_cfg_dir --> xxx.cfg
    [ rename(raw_dir) --> ] images_dir --> xxx.txt
    xxx.txt xxx.names --> xxx.data
    """
    def __init__(self,cfg,rename_tag_fun):
        self.cfg=cfg
        self.rename_tag=rename_tag_fun
        # ['excavator','loader','truck']
        self.class_names=[name.lower() for name in cfg.class_names]
        self.class_num=len(self.class_names)
        self.images_dir=cfg.images_dir
        self.save_cfg_dir=cfg.save_cfg_dir
        self.note=cfg.note
        self.overwrite=cfg.overwrite
        self.train_dir=cfg.train_dir
        os.chdir(self.train_dir)
        self.unknown_names=[]
        self.object_count={}
        self.object_count['img_file']=0
        self.object_count['xml_file']=0
        
    def update_object_count(self,name):
        if name in self.object_count.keys():
            self.object_count[name]+=1
        else:
            self.object_count[name]=1
            
    def generate_txt(self,raw_dir):
        """
        convert xml dataset to txt dataset
        for different self.class_names, we have different class
        self.images_dir dir for image and xml
        self.txt_dir dir for output txt annotations
        """
        files=get_files(raw_dir)
            
        suffix=('jpg','jpeg','bmp','png')
        img_files=[f for f in files if f.lower().endswith(suffix) and check_img(f)]
#         xml_files=glob.glob(os.path.join(in_path,'*','*.xml'))
#         img_files=[f for f in img_files if f.replace('jpg','xml') in xml_files]
        
        for img_f in trange(img_files):
            self.object_count['img_file']+=1
            xml_f=os.path.splitext(img_f)[0]+'.xml'
            if os.path.exists(xml_f):
                self.object_count['xml_file']+=1
                txt_file=os.path.join(self.images_dir,os.path.basename(xml_f).replace('.xml','.txt'))
                new_img_f=os.path.join(self.images_dir,os.path.basename(img_f))
                os.makedirs(os.path.dirname(new_img_f),exist_ok=True)
                if self.overwrite or not os.path.exists(new_img_f):
                    shutil.copy(img_f,new_img_f)
                assert check_img(new_img_f),'bad image {}-->{}'.format(img_f,new_img_f)
                
                os.makedirs(os.path.dirname(txt_file),exist_ok=True)
    #             print("convert {} to {}".format(new_xml_f,txt_file))
                self.convert2darknet_label(xml_f,txt_file,img_f)
                
    def rename(self,raw_dir):
        """
        raw_dir: the dir for raw label files
        problem: loss xml, bad image name
        result: rename image name and generate darknet format label
        todo: image without xml means no objects?
        """
        out_path=self.images_dir
        
        files=get_files(raw_dir)
        
        suffix=('jpg','jpeg','bmp','png')
        img_files=[f for f in files if f.lower().endswith(suffix) and check_img(f)]
#         xml_files=glob.glob(os.path.join(in_path,'*','*.xml'))
#         img_files=[f for f in img_files if f.replace('jpg','xml') in xml_files]
        
        # name image with {:06d}.jpg
        assert len(img_files)<999999
        idx=0
        for img_f in trange(img_files):
            self.object_count['img_file']+=1
            xml_f=os.path.splitext(img_f)[0]+'.xml'
            xml_f=xml_f.replace('JPEGImages','Annotations')
            
            # just check how to get xml file from image file
            img_suffix=os.path.splitext(img_f)[1]
            test_xml_f=img_f.replace(img_suffix,'.xml').replace('JPEGImages','Annotations')
            if os.path.exists(test_xml_f) and not os.path.exists(xml_f):
                warnings.warn('how to replace? {}'.format(img_f))
                
            if os.path.exists(xml_f):
                self.object_count['xml_file']+=1
                new_img_f=os.path.join(out_path,'{:06d}{}'.format(idx,img_suffix))
                new_xml_f=os.path.join(out_path,'{:06d}.xml'.format(idx))
                idx=idx+1
    #             print('copy {} to {}'.format(img_f,new_img_f))
                assert img_f!=new_img_f
                assert xml_f!=new_xml_f

                os.makedirs(os.path.dirname(new_img_f),exist_ok=True)
                if self.overwrite or not os.path.exists(new_img_f):
                    shutil.copy(img_f,new_img_f)
                if self.overwrite or not os.path.exists(new_xml_f):
                    shutil.copy(xml_f,new_xml_f)
                
                assert check_img(new_img_f),'bad image {}-->{} {}--> {}'.format(img_f,new_img_f,xml_f,new_xml_f)
                
                txt_file=new_xml_f.replace('/images/','/labels/').replace('.xml','.txt')
                os.makedirs(os.path.dirname(txt_file),exist_ok=True)
    #             print("convert {} to {}".format(new_xml_f,txt_file))
                self.convert2darknet_label(new_xml_f,txt_file,new_img_f)
                
    def convert2darknet_label(self,ann_file,out_file,img_file):
        write_file=None
        
        try:
            tree=ET.parse(ann_file)
        except Exception as e:
            print('cannot parse ann_file',ann_file,e)
            assert False
        
        objs = tree.findall('object')
        
        width=int(tree.find('size').find('width').text)
        height=int(tree.find('size').find('height').text)
        
        try:
            img=cv2.imread(img_file)
            img_h,img_w=img.shape[0:2]
            if width!=img_w or height!=img_h:
                warnings.warn('bad size for annotation file {}'.format(ann_file))
                print(height,width,img_h,img_w)
                height=img_h
                width=img_w
        except Exception as e:
            print(img_file,ann_file,e)
            if img is None:
                assert False
            
        assert width>0 and height>0,'for file {}'.format(ann_file)
        
        for obj in objs:
            name = obj.find('name')
            bndbox=obj.find('bndbox')
            x1=int(bndbox.find('xmin').text)
            y1=int(bndbox.find('ymin').text)
            x2=int(bndbox.find('xmax').text)
            y2=int(bndbox.find('ymax').text)
            
            assert width>=x2>x1 and height>=y2>y1,'width={}, height={} in {}'.format(width,height,ann_file)
            
            xc=(x1+x2)/width/2
            yc=(y1+y2)/height/2
            w=(x2-x1)/width
            h=(y2-y1)/height
            # ['excavator','loader','truck']
            tag=self.rename_tag(name.text)
            if tag in self.class_names:
                if write_file is None:
                    write_file=open(out_file,'w')
                idx=self.class_names.index(tag)
                write_line=' '.join([str(idx),str(xc),str(yc),str(w),str(h)])
                write_file.write(write_line+'\n')
                self.update_object_count(tag)
            else:
                if tag=='unknown' and name.text.lower() not in self.unknown_names:
                    self.unknown_names.append(name.text.lower())
                    print('unknown name',name.text,ann_file)

        if write_file is not None:
            write_file.close()
        
    def generate_files(self):
        """
        generate config files like xxx.data, xxx.txt
        """
        os.makedirs(self.save_cfg_dir,exist_ok=True)
        data_file=os.path.join(self.save_cfg_dir,self.note+'.data')
        train_file=os.path.join(self.save_cfg_dir,self.note+'_train.txt')
        valid_file=os.path.join(self.save_cfg_dir,self.note+'_valid.txt')
        name_file=os.path.join(self.save_cfg_dir,self.note+'.names')
        config_file=os.path.join(self.save_cfg_dir,self.note+'.cfg')
        with open(data_file,'w') as f:
            f.write('classes={}\n'.format(self.class_num))
            f.write('train={}\n'.format(train_file))
            f.write('valid={}\n'.format(valid_file))
            f.write('names={}\n'.format(name_file))
            f.write('backup={}\n'.format('backup/'))
            f.write('eval=coco\n')
        
        files=glob.glob(os.path.join(self.images_dir,'*.*'))
        suffix=('jpg','jpeg','bmp','png')
        img_files=[f for f in files if f.lower().endswith(suffix)]
        random.shuffle(img_files)
        
        N=len(img_files)*3//4
        train_imgs=img_files[0:N]
        valid_imgs=img_files[N:]
        print('train dataset size',len(train_imgs))
        print('valid dataset size',len(valid_imgs))
        print(self.object_count)
        with open(train_file,'w') as f:
            img_files=train_imgs
            for img_f in img_files:
                f.write(img_f+'\n')
                
        with open(valid_file,'w') as f:
            img_files=valid_imgs
            for img_f in img_files:
                f.write(img_f+'\n')
                
        with open(name_file,'w') as f:
            for n in self.class_names:
                f.write(n+'\n')

        env = Environment(loader=FileSystemLoader(self.save_cfg_dir))
        if self.cfg.transfer:
            template=env.get_template('yolov3_transfer.cfg.template')
        else:
            template = env.get_template('yolov3.cfg.template')     
        output = template.render(classes=self.class_num,filters=self.class_num*3+15)

        with open(config_file, 'w') as f:
            f.write(output)
        print('generate {} {} {} {}'.format(data_file,train_file,name_file,config_file))
        
        return data_file,config_file
    
    def train_yolov3(self,raw_dir):
        if self.cfg.rename_dataset:
            self.rename(raw_dir)
        else:
            self.generate_txt(raw_dir)
        
        data_file,cfg_file=self.generate_files()
        
        print("run the follow code to train\n")
        print("cd {}".format(self.train_dir))
        print("python train.py --data {data_file} --cfg {cfg_file} --notest --epoch 30 --nosave && mv weights/latest.pt weights/{note}.pt".format(data_file=data_file,cfg_file=cfg_file,note=self.note))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    now=datetime.now()
    parser.add_argument('--note', default='digger',help='name for config files and weight file')
    parser.add_argument('--class_num',default=1,type=int,help='class number for dataset')
    parser.add_argument('--save_cfg_dir', default='yzbx', help='dir for config files and template')
    parser.add_argument('--train_dir', default=os.path.expanduser('~/git/torchdet/model/yolov3'), help='directory for trainning code')
    parser.add_argument('--images_dir',default=None,help='directory to save reorder images/xml/txt')
    parser.add_argument('--raw_dir',default=os.path.expanduser('~/cvdataset/QingDao/digger'),help='directory for raw labeled images')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--rename_dataset',action='store_true')
    parser.add_argument('--transfer',action='store_true')
    args = parser.parse_args()
    
    
    cfg=edict()
    class_names=['excavator','truck','loader']
    cfg.class_names=class_names[0:args.class_num]
    cfg.note='_'.join([args.note,'cls'+str(args.class_num),now.strftime('%Y%m%d')])
    if args.images_dir is None:
        cfg.images_dir=os.path.join(os.path.expanduser('~/cvdataset/QingDao'),cfg.note)
    else:
        cfg.images_dir=args.images_dir
    
    assert cfg.images_dir!=args.raw_dir,'{}!={}'.format(cfg.images_dir,args.raw_dir)
    
    cfg.save_cfg_dir=args.save_cfg_dir
    cfg.train_dir=args.train_dir
    
    cfg.overwrite=args.overwrite
    cfg.rename_dataset=args.rename_dataset
    cfg.transfer=args.transfer
    
    pipeline=darknet_pipeline(cfg,rename_tag)
    pipeline.train_yolov3(args.raw_dir)