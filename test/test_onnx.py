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
import numpy as np
if __name__ == '__main__':
    model_files=glob.glob(os.path.join('/home/yzbx/logs/vgg11/lighter/**/*.pt'),recursive=True)

    assert len(model_files)>0
    model_files.sort()
    print(model_files)
    model_name='vgg11'

    pytorch_model=get_model(model_name,model_files[0])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for idx,model_path in enumerate(model_files):
        model=get_model(model_name,model_path)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rand_img=torch.rand((1,3,224,224)).to(device)
        outputs=model.forward(rand_img)

        torch.onnx.export(model,
                          rand_img,
                          str(idx)+'.onnx',
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          verbose=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input':{0:'batch_size'},
                                        'output':{0:'batch_size'}})

    onnx_model=onnx.load('0.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session=onnxruntime.InferenceSession('0.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#    ort_inputs={ort_session.get_inputs()[0].name:to_numpy(img)}
#    ort_outs=ort_session.run(None,ort_inputs)

#    print(ort_outs)

    cap=cv2.VideoCapture('dataset/smoke/FireSense/fire/pos/posVideo10.869.avi')

    if not cap.isOpened():
        assert False

    names=['normal','fire']
    while True:
        flag,frame=cap.read()
        if not flag:
            break

        print('frame mean',np.mean(frame,axis=(0,1)))

        np_mean=np.mean(simple_preprocess(frame,(224,224)),axis=(1,2))
        print('np mean 3d',np_mean)

        img=np.expand_dims(simple_preprocess(frame,(224,224)),0)
        np_mean=np.mean(img,axis=(2,3))
        print('np_mean 4d',np_mean)
        # pytorch result
        pytorch_input=torch.from_numpy(img).to(device)
        mean_input=torch.mean(pytorch_input,dim=[2,3])
        print('torch mean',mean_input)
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

        cv2.imshow('fire detection',frame)
        key=cv2.waitKey(30)
        if key==ord('q'):
            break
