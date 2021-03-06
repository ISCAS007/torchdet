# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
convert QingDao helmet dataset to darknet format
- https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
convert Chinese Code
- yzbxLib/tmp/config/yzbx_helmet_label.py
head and face detection dataset
- https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
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
from dataset.pipeline import darknet_pipeline

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
    if tag in ['Safety hat','helmet','安全帽','yellow','white','red','blue']:
        return 'helmet'
    elif tag in ['Safety belt','seatbelt','安全带']:
        return 'seatbelt'
    elif tag in ['person','人']:
        return 'people'
    elif tag in ['none']:
        return 'none'
    else:
        return 'unknown'

def rename_tag_color(tag):
    labels=['yellow','white','red','blue','orange','others']
    if tag in labels:
        return tag
    elif '-helmet' in tag or len(tag)==1:
        for l in labels:
            if tag[0] == l[0]:
                return l
        if tag in ['qt-helmet']:
            return 'others'
        else:
            return 'unknown'
    elif tag in ['none']:
        return tag
    else:
        return 'unknown'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    now=datetime.now()
    parser.add_argument('--note', default='helmet',help='name for config files and weight file')
    parser.add_argument('--class_num',default=2,type=int,help='class number for dataset')
    parser.add_argument('--save_cfg_dir', default='yzbx', help='dir for config files and template')
    parser.add_argument('--train_dir', default=os.path.expanduser('~/git/torchdet/model/yolov3'), help='directory for trainning code')
    parser.add_argument('--images_dir',default=None,help='directory to save reorder images/xml/txt')
    parser.add_argument('--raw_dir',default=os.path.expanduser('~/cvdataset/helmet/helmet'),help='directory for raw labeled images')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--rename_dataset',action='store_true')
    parser.add_argument('--spp',action='store_true')
    parser.add_argument('--color',action='store_true')
    parser.add_argument('--keep_image_without_label',action='store_true')

    args = parser.parse_args()


    cfg=edict()
    if args.color:
        class_names=['yellow','white','red','blue','orange','others','none']
    else:
        class_names=['helmet','none','people','seatbelt']
    cfg.class_names=class_names[0:args.class_num]
    cfg.note='_'.join([args.note,'cls'+str(args.class_num),now.strftime('%Y%m%d')])
    if args.images_dir is None:
        cfg.images_dir=os.path.join(os.path.expanduser('~/cvdataset/helmet'),cfg.note)
    else:
        cfg.images_dir=args.images_dir

    assert cfg.images_dir!=args.raw_dir,'{}!={}'.format(cfg.images_dir,args.raw_dir)
    cfg.save_cfg_dir=args.save_cfg_dir
    cfg.train_dir=args.train_dir
    cfg.overwrite=args.overwrite
    cfg.rename_dataset=args.rename_dataset
    cfg.spp=args.spp
    cfg.keep_image_without_label=args.keep_image_without_label

    if args.color:
        assert args.class_num==len(class_names)
        pipeline=darknet_pipeline(cfg,rename_tag_color)
    else:
        pipeline=darknet_pipeline(cfg,rename_tag)
    pipeline.train_yolov3(args.raw_dir)