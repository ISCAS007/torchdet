# -*- coding: utf-8 -*-
"""
view model/smoke/dataset_convert.py for detail
convert dataset to txt format, and organize them together
"""
import json
import os
import glob


class JsonClsDataset:
    def __init__(self,json_file,names):
        with open(json_file,'r') as f:
            cfg=json.load(f)

        self.names=names

        for name in names:
            for split in ['train','test']:
                img_files=[]
                for f in cfg[name]:
                    if os.path.basename(f).find(split)!=-1:
                        root_path=os.path.dirname(f)
                        with open(f,'r') as txt_file:
                            for line in txt_file.readlines():
                                line=line.strip()
                                if line !='':
                                    if os.path.exists(line):
                                        img_files.append(line)
                                    else:
                                        img_path=os.path.join(root_path,line)
                                        assert os.path.exists(img_path),img_path

                print(name,split,len(img_files))

if __name__ == '__main__':
    root_path='dataset/smoke'

    txt_files=glob.glob(os.path.join(root_path,'**','*.txt'),recursive=True)
    print(txt_files)

    dataset={}
    names=['fire','smoke','normal']
    for name in names:
        dataset[name]=[]
        for f in txt_files:
            base_name=os.path.basename(f)
            if base_name.find(name)!=-1:
                dataset[name].append(f)


    print(json.dumps(dataset))
    config_file=os.path.join(root_path,'dataset.json')
    with open(config_file,'w') as f:
        json.dump(dataset,f)

    JsonClsDataset(config_file,names)