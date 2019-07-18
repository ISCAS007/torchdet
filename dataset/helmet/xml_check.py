# -*- coding: utf-8 -*-

# unknown name 人 /home/yzbx/cvdataset/helmet/helmet/000000.xml
# unknown name 安全带 /home/yzbx/cvdataset/helmet/helmet/000008.xml

import glob
import os
import xml.etree.ElementTree as ET

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

root_dir=os.path.expanduser('~/cvdataset/helmet/old/helmet_github')
xml_files=glob.glob(os.path.join(root_dir,'**','*.xml'),recursive=True)
names=['安全帽', '人']
#xml_f='/home/yzbx/cvdataset/helmet/helmet/000000.xml'
for xml_f in xml_files:
    tree=ET.parse(xml_f)
    objs = tree.findall('object')
    for obj in objs:
        name = obj.find('name')
        tag=rename_tag(name.text)
        if tag=='unknown' and name.text not in names:
            print(name.text)
            names.append(name.text)