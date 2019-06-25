import argparse
import time
from sys import platform
import os
import sys
sys.path.append('./model/yolov3')
from model.yolov3.models import *
from model.yolov3.utils.datasets import *
from model.yolov3.utils.utils import *
from util.split_image import split_image,merge_image,yolov3_loadImages,yolov3_loadVideos
import numpy as np
import torch
import pickle

def merge_bbox(bboxes,target_size,origin_size,conf_thres=0.5,nms_thres=0.5):
    """
    use slide window technology
    bboxes: small image's detection results
    target_size: small image normed size
    origin_size: full image size
    """
    if isinstance(target_size,int):
        target_size=(target_size,target_size)
        
    h,w=origin_size[0:2]
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1

    merged_bbox=[]
    for i in range(h_num):
        for j in range(w_num):
            if i==h_num-1:
                h_end=h
            else:
                h_end=i*th+2*th
            
            if j==w_num-1:
                w_end=w
            else:
                w_end=j*tw+2*tw
            
            idx=i*w_num+j
            # image size in slide window technology >= target_size
            shape=(h_end-i*th,w_end-j*tw)
            offset=(th*i,tw*j)
            if bboxes[idx] is not None:
                det=bboxes[idx]
                if target_size!=shape:
                    # Rescale boxes from target size to slide window size
                    y_h_scale=shape[0]/target_size[0]
                    x_w_scale=shape[1]/target_size[1]
                    det[:,[0,2]] =(det[:,[0,2]]*x_w_scale).round()
                    det[:,[1,3]]=(det[:,[1,3]]*y_h_scale).round()
                    det[:,0:4]=det[:,0:4].clamp(min=0)
                det[:,:4]+=torch.tensor([offset[1],offset[0],offset[1],offset[0]]).to(det)

                merged_bbox.append(det)
                    
    merged_bbox=torch.cat(merged_bbox,dim=0)
    nms_merged_bbox=torch.zeros_like(merge_bbox)
    nms_merged_bbox[:,0]=(merged_bbox[:,0]+merged_bbox[:,2])/2
    nms_merged_bbox[:,1]=(merged_bbox[:,1]+merged_bbox[:,3])/2
    nms_merged_bbox[:,2]=(merged_bbox[:,2]-merged_bbox[:,0])
    nms_merged_bbox[:,3]=(merged_bbox[:,3]+merged_bbox[:,1])
    # nms input format [center x,center y,w,h]
    # nms output format [x1,y1,x2,y2]
    out_nms_merged_bbox = non_max_suppression([nms_merged_bbox], conf_thres, nms_thres)[0]

    data={
        'bboxes':bboxes,
        'target_size':target_size,
        'origin_size':origin_size,
        'merged_bbox':merged_bbox,
        'nms_merged_bbox':nms_merged_bbox,
        'out_nms_merged_bbox':out_nms_merged_bbox,
    }
    with open('data.pkl','wb') as f:
        pickle.dump(data,f)
    return out_nms_merged_bbox

def filter_label(det,classes,device):
    if det is not None:
        det_idx=[]
        for c in det[:,-1]:
            if classes[int(c)] not in ['car','person','bicycle','motorbike','truck']:
                print('filter out',classes[int(c)])
                det_idx.append(0)
            else:
                det_idx.append(1)
        if np.any(det_idx):
            det=det[torch.from_numpy(np.array(det_idx)).to(device).eq(1),:]
        else:
            det=None


def detect(
        cfg,
        data_cfg,
        weights,
        input_folder='data/samples',  # input folder
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_result=True,
        load_video=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (416, 416)  # onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    if load_video:
        current_video_path=None
        writer=None
        dataloader = yolov3_loadVideos(input_folder,img_size=img_size)
    else:
        dataloader = yolov3_loadImages(input_folder, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, resize_imgs, split_imgs,origin_img) in enumerate(dataloader):
        t = time.time()
        save_path = path.replace(input_folder,output)
        if load_video:
            save_path=save_path.replace(save_path[-4:],'.mp4')
        
        assert save_path!=path
        
        #batch process
        if len(resize_imgs)>0:
            batch_imgs=torch.stack([torch.from_numpy(img) for img in resize_imgs]).to(device)
            batch_pred,_=model(batch_imgs)
            #batch_det is a detection result list for img in batch_imgs
            batch_det=non_max_suppression(batch_pred, conf_thres, nms_thres)

            if batch_det is not None:
                merged_det=merge_bbox(batch_det,img_size,origin_img.shape[:2],conf_thres,nms_thres)
                draw_origin_img=origin_img.copy()
                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in merged_det:
                    # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, draw_origin_img, label=label, color=colors[int(cls)])

                print('merget_det',merged_det.shape)
                filename='merge_bbox'+str(int(time.time()))+'.jpg'
                cv2.imwrite(filename,draw_origin_img)

        draw_imgs=[]
        for resize_img,split_img in zip(resize_imgs,split_imgs):
            # Get detections
            img = torch.from_numpy(resize_img).unsqueeze(0).to(device)
            pred, _ = model(img)
            det = non_max_suppression(pred, conf_thres, nms_thres)[0]

            #filter_label(det,classes,device)

            if det is not None and len(det) > 0:
                # Rescale boxes from 416 to true image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], split_img.shape).round()
    
                # Print results to screen
                print('%gx%g ' % img.shape[2:], end='')  # print image size
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')
    
                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
    
                    # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, split_img, label=label, color=colors[int(cls)])
                    
            draw_imgs.append(split_img)
        draw_img=merge_image(draw_imgs,img_size,origin_img.shape)
        print('Done. (%.3fs)' % (time.time() - t))
        filename='merge_img'+str(int(time.time()))+'.jpg'
        cv2.imwrite(filename,draw_img)

        if i >= 0:
            break
        
        if load_video:  # Show live webcam
            if save_result:
                if current_video_path==save_path:
                    pass
                else:
                    if current_video_path is not None:
                        writer.release()
                    current_video_path=save_path
                    codec = cv2.VideoWriter_fourcc(*"mp4v")
                    #fps = reader.get(cv2.CAP_PROP_FPS)
                    #width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps=30
                    height,width,_=draw_img.shape
                    
                    print('save video to',save_path)
                    assert save_path!=path
                    dirname=os.path.dirname(save_path)
                    os.makedirs(dirname,exist_ok=True)
                    writer = cv2.VideoWriter(save_path, codec, fps, (width, height))
                
                writer.write(draw_img)
                
        else:
            if save_result:  # Save image with detections
                cv2.imwrite(save_path, draw_img)

    if save_result:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    cfg_path='model/yolov3/yzbx'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=os.path.join(cfg_path,'yolov3-spp.cfg'), help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default=os.path.join(cfg_path,'coco.data'), help='coco.data file path')
    parser.add_argument('--weights', type=str, default=os.path.join(cfg_path,'yolov3-spp.weights'), help='path to weights file')
    parser.add_argument('--input_folder', type=str, default='dataset/demo/crossroad', help='path to input_folder')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--load_video',default=False,action='store_true',help='load video or image')
    parser.add_argument('--save_result',default=False,action='store_true',help='load video or image')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            input_folder=opt.input_folder,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            load_video=opt.load_video,
            save_result=opt.save_result
        )
