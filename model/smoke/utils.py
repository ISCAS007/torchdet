# -*- coding: utf-8 -*-

import onnx
import torch

def to_onnx(model,save_path):
    model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    rand_img=torch.rand((1,3,224,224)).to(device)
    outputs=model.forward(rand_img)

    torch.onnx.export(model,
                      rand_img,
                      save_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},
                                    'output':{0:'batch_size'}})

    onnx_model=onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
