# -*- coding: utf-8 -*-

import sys
sys.path.append('./model/Detectron')

from tools.infer_simple import *
from util.split_image import My_VideoWriter,yolov3_loadVideos,merge_image

def my_detectron(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]
    
    input_image_size=[600,600]
    dataloader = yolov3_loadVideos(args.im_or_folder,img_size=input_image_size,preprocess=False)
    videowriter=My_VideoWriter()
    for i, (path, resize_imgs, split_imgs,origin_img) in enumerate(dataloader):
#    for i, im_name in enumerate(im_list):
        out_path = path.replace(args.im_or_folder,args.output_dir)
        assert out_path!=path

        logger.info('Processing {} -> {} {}'.format(path, out_path,i))
        
        draw_imgs=[]
        for resize_img,split_img in zip(resize_imgs,split_imgs):
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, resize_img, None, timers=timers
                )
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            if i == 0:
                logger.info(
                    ' \ Note: inference on the first image will be slower than the '
                    'rest (caches and auto-tuning need to warm up)'
                )
            
            draw_img=vis_utils.vis_one_image_opencv(resize_img,cls_boxes,show_box=True,show_class=True)
            draw_imgs.append(draw_img)
        output_image=merge_image(draw_imgs,input_image_size,origin_img.shape)
        videowriter.write(out_path,output_image)
    videowriter.release()
        
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    my_detectron(args)