# -*- coding: utf-8 -*-
import torch
from torchvision.models import vgg11,vgg13
import torch.utils.model_zoo as model_zoo
import cv2
import os
import numpy as np
import glob
from model.smoke.dataset import simple_preprocess

def get_model(model_name,model_path):
    model=globals()[model_name](pretrained=False,num_classes=2)

    if os.path.isdir(model_path):
        model_path_list=glob.glob(os.path.join(model_path,'**','*.pt'),recursive=True)
        assert len(model_path_list)>0
        model_path=model_path_list[-1]

        print('use the first from',model_path_list)

    model.load_state_dict(torch.load(model_path).state_dict())
    model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
if __name__ == '__main__':
    model_name='vgg11'
#    model_path='/home/yzbx/tmp/logs/torchdet/vgg11/all/all0802/2019-08-02___19-52-24/model_100.pt'
#    model_path='logs/vgg11/one/2019-12-04___16-23-05/model_30.pt'
#    model_path='logs/vgg11/lighter0.5/2019-12-04___17-58-03/model_30.pt'
#    model_path='logs/vgg11/lighter1.0/2019-12-04___19-03-32/model_30.pt'
#    model_path='/home/yzbx/logs/vgg11/lighter/lighter0.0/2019-12-05___14-58-56/model_30.pt'
#    model_path='/home/yzbx/logs/vgg11/lighter/lighter0.5/2019-12-05___20-46-49/model_30.pt'
    model_path='/home/yzbx/logs/vgg11/lighter/lighter0.4'

    #model_config_file='/home/yzbx/tmp/logs/torchdet/vgg11/all/all0802/2019-08-02___19-52-24/config.txt'

    model=get_model(model_name,model_path)

    video_name='dataset/smoke/FireSense/fire/pos/posVideo10.869.avi'
#    video_name='/home/yzbx/git/qd/templates/coal_fire_demo001.mp4'
#    video_name='/media/sdb/ISCAS_Dataset/driver_videos/抽烟1.h264'
#    video_name='/media/sdb/ISCAS_Dataset/driver_videos/打电话1.h264'
    img_size=(224,224)
    names=['normal','fire']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap=cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise Exception('cannot open video {}'.format(video_name))

    save_video_name=os.path.join(os.path.dirname(video_name),'out_'+os.path.basename(video_name))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    fps=30
    writer=None
    while True:
        flag,frame=cap.read()
        if flag:
            img=simple_preprocess(frame,img_size)
            inputs=torch.unsqueeze(torch.from_numpy(img),dim=0).to(device)

            print('frame mean',np.mean(frame,axis=(0,1)))
            np_mean=np.mean(img,axis=(1,2))
            print('np mean',np_mean)
            mean_input=torch.mean(inputs,dim=[2,3])
            print('torch mean',mean_input)

            outputs=model.forward(inputs)
            result=torch.softmax(outputs,dim=1).data.cpu().numpy()
            result=np.squeeze(result)

            text=names[np.argmax(result)]
            if text==names[1]:
                color=(0,0,255)
            else:
                color=(255,0,0)

            print(text,max(result))
            # convert image to [height width channel] format
            fontScale=max(1,frame.shape[1]//448)
            thickness=max(1,frame.shape[1]//112)
            frame=cv2.putText(frame, text+' %0.2f'%(max(result)) , (50,50), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness)

            cv2.imshow('fire detection',frame)
            key=cv2.waitKey(30)
            if key==ord('q'):
                break

            if writer is None:
                height,width=frame.shape[0:2]
                os.makedirs(os.path.dirname(save_video_name),exist_ok=True)
                writer=cv2.VideoWriter(save_video_name,
                                codec, fps,
                                (width, height))
            writer.write(frame)
        else:
            break
    writer.release()