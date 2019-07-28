# -*- coding: utf-8 -*-
import os
import unittest
import torch
from model.multi_dataset_model.cls_seg import Config,get_cls_dataloader,get_seg_dataloader

def convertCityscapesIdx(label):
    ids = [
    7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    reIndexed_label=torch.zeros_like(label)+255
    for idx,id in enumerate(ids):
        reIndexed_label=torch.where(label==id,torch.tensor(idx),reIndexed_label)

    return reIndexed_label

class TestDataset(unittest.TestCase):
    def test_cls_dataset(self):
        config=Config.get_config()
        config.root_path=os.path.join('~/cvdataset/places365_standard')

        loader=get_cls_dataloader(config,'train')

        for data in loader:
            for d in data:
                print(d.shape,torch.min(d),torch.max(d),d.dtype)

            break

        self.assertTrue(True)

    def test_seg_dataset(self):
        config=Config.get_config()
        loader=get_seg_dataloader(config,'train')


        for data in loader:
            for d in data:
                print(d.shape,torch.min(d),torch.max(d),d.dtype)

            d=convertCityscapesIdx(data[1])
            print(d.shape,torch.min(d),torch.max(d),d.dtype)

            x1,y1=torch.unique(data[1],return_counts=True)
            x2,y2=torch.unique(d,return_counts=True)
            print(x1,y1)
            print(x2,y2)
            break

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
