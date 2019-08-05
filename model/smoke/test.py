# -*- coding: utf-8 -*-
import torch
from torchvision.models import vgg11,vgg13
import torch.utils.model_zoo as model_zoo
import cv2
import numpy as np
from model.smoke.dataset import simple_preprocess
def get_model(model_name,model_path):
    model=globals()[model_name](pretrained=False,num_classes=2)

    model.load_state_dict(torch.load(model_path).state_dict())
    model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
if __name__ == '__main__':
    model_name='vgg11'
    model_path='/home/yzbx/tmp/logs/torchdet/vgg11/all/all0802/2019-08-02___19-52-24/model_100.pt'
    model_config_file='/home/yzbx/tmp/logs/torchdet/vgg11/all/all0802/2019-08-02___19-52-24/config.txt'

    model=get_model(model_name,model_path)

    video_name='/media/sdb/ISCAS_Dataset/driver_videos/抽烟1.h264'
#    video_name='/media/sdb/ISCAS_Dataset/driver_videos/打电话1.h264'
    img_size=(224,224)
    names=['normal','smoke']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap=cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise Exception('cannot open video {}'.format(video_name))

    while True:
        flag,img=cap.read()
        if flag:
            img=simple_preprocess(img,img_size)
            inputs=torch.unsqueeze(torch.from_numpy(img),dim=0).to(device)
            outputs=model.forward(inputs)
            result=torch.softmax(outputs,dim=1).data.cpu().numpy()
            result=np.squeeze(result)
            print(names[np.argmax(result)])
        else:
            break