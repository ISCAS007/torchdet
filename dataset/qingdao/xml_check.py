# -*- coding: utf-8 -*-
import os
import glob
import xml.etree.ElementTree as ET

raw_dir='/media/sdb/ISCAS_Dataset/QingDao/digger/20190712'
raw_dir='/media/sdb/ISCAS_Dataset/QingDao/digger_cls3_0712'
xmls=glob.glob(os.path.join(raw_dir,'*.xml'))

assert len(xmls)>0
print(len(xmls))
for f in xmls:
    try:
        tree=ET.parse(f)
    except Exception as e:
        print(f,e)
        assert False