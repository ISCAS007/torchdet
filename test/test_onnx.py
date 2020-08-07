# -*- coding: utf-8 -*-

import glob
import os
from model.smoke.test import get_model
import torch
import onnx
import onnxruntime
import cv2
from model.smoke.dataset import simple_preprocess
from scipy.special import softmax
from model.smoke.utils import to_onnx
import numpy as np
import sys

if __name__ == '__main__':
    note='json'
    target='smoke'
    model_files=glob.glob(os.path.join('/home/yzbx/logs/vgg11/{}/{}/**/*.pt'.format(target,note)),recursive=True)

    assert len(model_files)>0
    model_files.sort()
    print(model_files)
    model_name='vgg11'

    pytorch_model=get_model(model_name,model_files[0])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for idx,model_path in enumerate(model_files):
        model=get_model(model_name,model_path)
        to_onnx(model,note+str(idx)+'.onnx')

    onnx_model=onnx.load(note+'0.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session=onnxruntime.InferenceSession(note+'0.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#    ort_inputs={ort_session.get_inputs()[0].name:to_numpy(img)}
#    ort_outs=ort_session.run(None,ort_inputs)

#    print(ort_outs)

    if len(sys.argv)>1:
        video=sys.argv[1]
    else:
        if target=='fire':
            video='dataset/smoke/FireSense/fire/pos/posVideo10.869.avi'
        else:
            video='dataset/smoke/FireSense/smoke/pos/testpos01.817.avi'

    cap=cv2.VideoCapture(video)

    if not cap.isOpened():
        assert False,video

    names=['normal',target]
    while True:
        flag,frame=cap.read()
        if not flag:
            break

        img=np.expand_dims(simple_preprocess(frame,(224,224)),0)

        # pytorch result
        pytorch_input=torch.from_numpy(img).to(device)
        pytorch_output=pytorch_model.forward(pytorch_input)
        pytorch_result=np.squeeze(torch.softmax(pytorch_output,dim=1).data.cpu().numpy())

        # onnx result
        ort_inputs={ort_session.get_inputs()[0].name:img}
        result=softmax(np.squeeze(ort_session.run(None,ort_inputs)))

        print(result,pytorch_result)

        text=names[np.argmax(result)]
        if text==names[1]:
            color=(0,0,255)
        else:
            color=(255,0,0)

        print(text,result)
        # convert image to [height width channel] format
        fontScale=max(1,frame.shape[1]//448)
        thickness=max(1,frame.shape[1]//112)
        frame=cv2.putText(frame, text+' %0.2f'%(max(result)) , (50,50), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness)

        cv2.imshow('{} detection'.format(target),frame)
        key=cv2.waitKey(30)
        if key==ord('q'):
            break
