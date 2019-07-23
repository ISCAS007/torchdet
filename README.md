# torchdet
pytorch1.0 for object detection with open source model

## split image to detect small object
- remove bounding box on the boundary for each small image
- each small image >= 2x2 grid = target image size

![](dataset/vis/split_image.png)

## detection reference
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/ultralytics/yolov3
```
source activate pytorch1.0
cd $yolov3_dir
python detect.py --cfg yolov3.cfg --weights yolov3.weights --images dataset/demo
```
- https://github.com/facebookresearch/Detectron
```
source activate detectron
cd $detectron_dir
python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /tmp/detectron-visualizations \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo
# download model in /tmp/detectron-download-cache
# output image in /tmp/detectron-visualizations
```
- https://github.com/yhenon/pytorch-retinanet
- https://github.com/open-mmlab/mmdetection

## segmentation reference
- https://github.com/lingtengqiu/Deeperlab-pytorch

## dataset

| dataset name | image | object | category |
| - | - | - | - |
| VOC07 | 5k | 12k | 20 |
| VOC12 | 11k | 27k | 20 |
| ILSVRC | 517k | 534k | 200 |
| MS-COCO 17 | 164k | 897k | 80 |
| OID-2018 | 1910k | 15440k | 600 |

## requirement
- model/smoke/smoke_cls.py
- model/boundary.py
- model/overlap.py

