# -*- coding: utf-8 -*-
import os
import glob
import xml.etree.ElementTree as ET

root_dir=os.path.expanduser('/media/sdb/ISCAS_Dataset/QingDao/digger')
xmls=glob.glob(os.path.join(root_dir,'**','*.xml'),recursive=True)

assert len(xmls)>0
print(len(xmls))
#for f in xmls:
#    try:
#        tree=ET.parse(f)
#    except Exception as e:
#        print(f,e)
#        xmls.remove(f)
        
def rename_tag(tag):
    if tag in ['Excavating','excavator']:
        return 'excavator'
    elif tag in ['B-Truck','S-Truck','Truck']:
        return 'truck'
    elif tag in ['loader']:
        return 'loader'
    else:
        return 'unknown'


names=[]
#xml_f='/home/yzbx/cvdataset/helmet/helmet/000000.xml'
for xml_f in xmls:
    try:
        tree=ET.parse(xml_f)
    except:
        pass
    else:
        objs = tree.findall('object')
        for obj in objs:
            name = obj.find('name')
            tag=rename_tag(name.text)
            if tag=='unknown' and name.text not in names:
                print(name.text)
                names.append(name.text)
            
print(names)