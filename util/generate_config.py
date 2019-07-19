from jinja2 import Environment, FileSystemLoader
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num',default=1,type=int,help='class number for dataset')
    parser.add_argument('--save_dir',default='doc',help='class number for dataset')
    parser.add_argument('--template_path',default='doc/yolov3-spp.cfg.template',help='template path for yolov3.cfg.template yolov3-spp.cfg.template')
    
    args = parser.parse_args()
    
    env = Environment(loader=FileSystemLoader('.'))
    template=env.get_template(args.template_path)  
    output = template.render(classes=args.class_num,filters=args.class_num*3+15)
    
    if args.template_path.find('yolov3-spp')>=0:
        base_cfg='yolov3-spp'
    else:
        base_cfg='yolov3'
    config_file=os.path.join(args.save_dir,'_'.join([base_cfg,'cls'+str(args.class_num)+'.cfg']))
    with open(config_file, 'w') as f:
        f.write(output)