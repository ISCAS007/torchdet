# -*- coding: utf-8 -*-
"""
convert QingDao digger dataset to darknet format
- https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
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

class darknet_pipeline():
    """
    ## pipeline 1
    xxx.names --> class_num --> xxx.cfg
    self.save_cfg_dir --> xxx.cfg
    [ rename(raw_dir) --> ] images_dir --> xxx.txt
    xxx.txt xxx.names --> xxx.data
    """
    def __init__(self,cfg):
        self.cfg=cfg
        # ['excavator','loader','truck']
        self.class_names=[name.lower() for name in cfg.class_names]
        self.class_num=len(self.class_names)
        self.images_dir=cfg.images_dir
        self.save_cfg_dir=cfg.save_cfg_dir
        self.note=cfg.note
        self.train_dir=cfg.train_dir
        os.chdir(self.train_dir)
        
    def rename(self,raw_dir):
        """
        raw_dir: the dir for raw label files
        problem: loss xml, bad image name
        result: rename image name and generate darknet format label
        todo: image without xml means no objects?
        """
        in_path=raw_dir
        out_path=self.images_dir
        
        files=glob.glob(os.path.join(in_path,'*','*.*'),recursive=True)
        suffix=('jpg','jpeg','bmp','png')
        img_files=[f for f in files if f.lower().endswith(suffix)]
#         xml_files=glob.glob(os.path.join(in_path,'*','*.xml'))
#         img_files=[f for f in img_files if f.replace('jpg','xml') in xml_files]
        
        # name image with {:06d}.jpg
        assert len(img_files)<999999
        idx=0
        for img_f in img_files:
            xml_f=img_f.replace('jpg','xml')
            if os.path.exists(xml_f):
                new_img_f=os.path.join(out_path,'{:06d}.jpg'.format(idx))
                new_xml_f=os.path.join(out_path,'{:06d}.xml'.format(idx))
                idx=idx+1
    #             print('copy {} to {}'.format(img_f,new_img_f))
                assert img_f!=new_img_f
                assert xml_f!=new_xml_f

                os.makedirs(os.path.dirname(new_img_f),exist_ok=True)
                if not os.path.exists(new_img_f):
                    shutil.copy(img_f,new_img_f)
                if not os.path.exists(new_xml_f):
                    shutil.copy(xml_f,new_xml_f)

                yolov3_file=new_xml_f.replace('/images/','/labels/').replace('.xml','.txt')
                os.makedirs(os.path.dirname(yolov3_file),exist_ok=True)
    #             print("convert {} to {}".format(new_xml_f,txt_file))
                self.convert2darknet_label(new_xml_f,yolov3_file)

                darknet_file=new_xml_f.replace('/images/','/labels/').replace('.xml','.txt')

    def convert2darknet_label(self,ann_file,out_file):
        write_file=None

        tree=ET.parse(ann_file)
        objs = tree.findall('object')
        for obj in objs:
            name = obj.find('name')
            bndbox=obj.find('bndbox')
            x1=int(bndbox.find('xmin').text)
            y1=int(bndbox.find('ymin').text)
            x2=int(bndbox.find('xmax').text)
            y2=int(bndbox.find('ymax').text)
            width=int(tree.find('size').find('width').text)
            height=int(tree.find('size').find('height').text)

            assert width>0 and height>0
            assert width>=x2>x1 and height>=y2>y1,'width={}, height={} in {}'.format(width,height,ann_file)
            xc=(x1+x2)/width/2
            yc=(y1+y2)/height/2
            w=(x2-x1)/width
            h=(y2-y1)/height
            # ['excavator','loader','truck']
            if name.text.lower() in self.class_names:
                if write_file is None:
                    write_file=open(out_file,'w')
                idx=self.class_names.index(name.text.lower())
                write_line=' '.join([str(idx),str(xc),str(yc),str(w),str(h)])
                write_file.write(write_line+'\n')
            else:
                print('unknown name',name.text)
        
        if write_file is not None:
            write_file.close()
        
    def generate_files(self):
        """
        generate config files like xxx.data, xxx.txt
        """
        os.makedirs(self.save_cfg_dir,exist_ok=True)
        data_file=os.path.join(self.save_cfg_dir,self.note+'.data')
        train_file=os.path.join(self.save_cfg_dir,self.note+'.txt')
        valid_file=train_file
        name_file=os.path.join(self.save_cfg_dir,self.note+'.names')
        config_file=os.path.join(self.save_cfg_dir,self.note+'.cfg')
        with open(data_file,'w') as f:
            f.write('classes={}\n'.format(self.class_num))
            f.write('train={}\n'.format(train_file))
            f.write('valid={}\n'.format(valid_file))
            f.write('names={}\n'.format(name_file))
            f.write('backup={}\n'.format('backup/'))
            f.write('eval=coco\n')
        
        
        with open(train_file,'w') as f:
            img_files=glob.glob(os.path.join(self.images_dir,"*.jpg"))
            for img_f in img_files:
                f.write(img_f+'\n')
                
        with open(name_file,'w') as f:
            for n in self.class_names:
                f.write(n+'\n')

        env = Environment(loader=FileSystemLoader(self.save_cfg_dir))
        template = env.get_template('yolov3.cfg.template')     
        output = template.render(classes=self.class_num,filters=self.class_num*3+15)

        with open(config_file, 'w') as f:
            f.write(output)
        print('generate {} {} {} {}'.format(data_file,train_file,name_file,config_file))
        
        return data_file,config_file
    
    def train_yolov3(self,raw_dir):
        self.rename(raw_dir)
        data_file,cfg_file=self.generate_files()
        
        print("run the follow code to train\n")
        print("cd {}".format(self.train_dir))
        print("python train.py --data {data_file} --cfg {cfg_file} --notest --epoch 30 --nosave && mv weights/latest.pt weights/{note}.pt".format(data_file=data_file,cfg_file=cfg_file,note=self.note))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    now=datetime.now()
    parser.add_argument('--note', default='digger'+now.strftime('%Y%m%d'),help='name for config files and weight file')
    parser.add_argument('--save_cfg_dir', default='yzbx', help='dir for config files and template')
    parser.add_argument('--train_dir', default='/home/yzbx/git/torchdet/model/yolov3', help='directory for trainning code')
    parser.add_argument('--images_dir',default=None,help='directory to save reorder images/xml/txt')
    parser.add_argument('--raw_dir',default='/media/sdb/ISCAS_Dataset/QingDao/digger',help='directory for raw labeled images')
    args = parser.parse_args()
    
    
    cfg=edict()
    cfg.class_names=['excavator','truck','loader']
    if args.images_dir is None:
        cfg.images_dir=os.path.join(os.path.dirname(args.raw_dir),args.note)
    else:
        cfg.images_dir=args.images_dir
    cfg.save_cfg_dir=args.save_cfg_dir
    cfg.train_dir=args.train_dir
    cfg.note=args.note
    
    pipeline=darknet_pipeline(cfg)
    pipeline.train_yolov3(args.raw_dir)