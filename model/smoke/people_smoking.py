# -*- coding: utf-8 -*-

import cv2
import torch.utils.data as td
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from easydict import EasyDict as edict
import time
import json
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import warnings
import argparse
from torchvision import transforms

def check_img(img_f):
    try:
        img=cv2.imread(img_f)
        img_size=(224,224)
        cv2.resize(img,img_size,interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        warnings.warn('bad image {} with exception {}'.format(img_f,e))
        return False
    else:
        if img is None:
            warnings.warn('bad image {}'.format(img_f))
            return False
        else:
            return True

class PosNegClsDataset:
    def __init__(self,root_path):
        self.pos_files=glob.glob(os.path.join(root_path,'pos','**','*.*'),recursive=True)
        self.neg_files=glob.glob(os.path.join(root_path,'neg','**','*.*'),recursive=True)

        img_suffix=('jpg','jpeg','bmp','png')
        self.pos_files=[f for f in self.pos_files if f.lower().endswith(img_suffix) and check_img(f)]
        self.neg_files=[f for f in self.neg_files if f.lower().endswith(img_suffix) and check_img(f)]
        self.pos_files.sort()
        self.neg_files.sort()

        assert len(self.pos_files)>0,'positive sample images cannot be empty'
        assert len(self.neg_files)>0,'negative sample images cannot be empty'

        self.names=['normal','smoking']

    def get_train_val(self,test_size=0.33,random_state=25):
        x=self.pos_files+self.neg_files
        # index 1 for pos, 0 for neg
        y=[1]*len(self.pos_files)+[0]*len(self.neg_files)

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
        return x_train,x_test,y_train,y_test

def simple_preprocess(image,img_size):
    # Padded resize
    img=cv2.resize(image,tuple(img_size),interpolation=cv2.INTER_LINEAR)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


class cls_dataset(td.Dataset):
    def __init__(self,config,split='train'):
        super().__init__()
        self.config=config
        self.split=split
        self.img_size=config.img_size
        image_dataset=PosNegClsDataset(config.root_path)

        x_train,x_test,y_train,y_test=image_dataset.get_train_val()
        if split=='train':
            self.x=x_train
            self.y=y_train
        elif split in ['val','test']:
            self.x=x_test
            self.y=y_test
        else:
            assert False

        self.count=0
        self.size=len(self.x)
        assert self.size>0,print(config)
        assert len(self.x)==len(self.y),'x and y must be the same size'
        print('{} dataset {} size={}'.format(config.root_path,split,self.size))

    def __iter__(self):
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        if self.count>=self.size:
            raise StopIteration

        path=self.x[self.count]
        origin_img=cv2.imread(path)
        post_img=self.preprocess(origin_img)
        label=self.y[self.count]

        self.count+=1
        return {'path':path,
                'post_img':post_img,
                'origin_img':origin_img,
                'label':label}

    def __getitem__(self,idx):
        self.count=idx

        data=self.__next__()
        new_data={}
        new_data['post_img']=data['post_img']
        new_data['label']=data['label']
        return new_data

    def preprocess(self,pre_img):
        return simple_preprocess(pre_img,self.img_size)

class cls_metric():
    def __init__(self):
        self.reset()

    def update(self,prediction,label,loss):
        self.tp+=torch.sum(prediction==label)
        self.count+=len(label)
        self.loss+=loss

    def get_metric(self):
        assert torch.is_tensor(self.tp)
        assert torch.is_tensor(self.loss)
        assert self.count>0
        return self.tp.item()/self.count,self.loss.item()/self.count

    def reset(self):
        self.tp=0
        self.count=0
        self.loss=0

class trainer:
    def __init__(self,config):
        self.config=config

        ## set dataset
        self.dataset={}
        if config.app in ['train']:
            for split in ['train','val']:
                batch_size=config.batch_size if split=='train' else 1
                shuffle=True if split=='train' else False
                drop_last=True if split=='train' else False
                dataset=cls_dataset(config,split)

                print(split,'dataset size is',len(dataset))
                self.dataset[split]=td.DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self.get_model()
        self.model.to(self.device)

        if config.app in ['train']:
            self.loss_fn=torch.nn.CrossEntropyLoss()
            ## set optimizer
            optimizer_params = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
            self.optimizer=torch.optim.Adam(optimizer_params,lr=config.lr)
            self.metric=cls_metric()
            time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
            self.log_dir = os.path.join(config.log_dir, config.model_name, config.dataset_name,
                                        config.note, time_str)

            self.writer=self.init_writer(config,self.log_dir)
        elif config.app in ['test']:
            model_path=self.config.load_model_path
            if not os.path.exists(model_path):
                print('model path',model_path,'not exist')
                return 0

            if os.path.isdir(model_path):
                model_path_list=glob.glob(os.path.join(model_path,'**','*.pt'),recursive=True)
                if len(model_path_list)==0:
                    print('cannot find weight file in directory',model_path)
                    return 0

                print(model_path_list)
                model_path=model_path_list[0]
                print('auto select model',model_path)

            self.model.load_state_dict(torch.load(model_path).state_dict())
            self.model.eval()
        elif config.app in ['vis']:
            pass
        else:
            assert False


    def get_model(self):
        model_urls = {
            'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
            'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
            'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
            'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
            'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        }

        from torchvision.models import vgg11,vgg13
        import torch.utils.model_zoo as model_zoo

        model=locals()[self.config.model_name](pretrained=False,num_classes=2)

        load_state_dict=model_zoo.load_url(model_urls[self.config.model_name])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in load_state_dict.items() if k.find('classifier.6')==-1}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        count=0
        for name,m in model.features.named_children():
            count+=1

        max_freeze_index=count*self.config.freeze_ratio
        idx=0
        for name,m in model.features.named_children():
            if idx<max_freeze_index:
                for param in m.parameters():
                    param.requires_grad = False

            idx+=1

        return model

    def init_writer(self,config, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        config_str = json.dumps(config, indent=2, sort_keys=True).replace(
            '\n', '\n\n').replace('  ', '\t')
        writer.add_text(tag='config', text_string=config_str)

        # write config to config.txt
        config_path = os.path.join(log_dir, 'config.txt')
        config_file = open(config_path, 'w')
        json.dump(config, config_file, sort_keys=True)
        config_file.close()

        return writer

    def train(self,epoch,split='train'):
        if split=='train':
            self.model.train()
        else:
            self.model.eval()

        self.metric.reset()

        tqdm_step = tqdm(self.dataset[split], desc='step', leave=False)
        assert len(self.dataset[split])>0
        for i,(data) in enumerate(tqdm_step):
            if split=='train':
                self.optimizer.zero_grad()
            inputs=data['post_img'].to(self.device)
            label=data['label'].to(self.device)
            outputs=self.model.forward(inputs)

            loss=self.loss_fn(outputs,label)

            if split=='train':
                loss.backward()
                self.optimizer.step()

            tqdm_step.set_postfix(loss=loss.item())

            predictions=torch.argmax(outputs,dim=1)
            self.metric.update(predictions,label,loss)

        acc,loss=self.metric.get_metric()
        self.writer.add_scalar('{}/loss'.format(split),loss,epoch)
        self.writer.add_scalar('{}/acc'.format(split),acc,epoch)

    def validation(self,epoch):
        self.train(epoch,split='val')

    def train_val(self):
        tqdm_epoch = trange(self.config.epoch, desc='epoch', leave=True)
        for epoch in tqdm_epoch:
            self.train(epoch)
            acc,loss=self.metric.get_metric()
            tqdm_epoch.set_postfix(train_acc=acc,loss=loss)

            with torch.no_grad():
                self.validation(epoch)
            acc,loss=self.metric.get_metric()
            tqdm_epoch.set_postfix(val_acc=acc,loss=loss)

        self.writer.close()

        if self.config.save_model:
            save_model_path=os.path.join(self.log_dir,'model_{}.pt'.format(self.config.epoch))
            print('save model to',save_model_path)
            torch.save(self.model,save_model_path)

    def test(self,image):
        assert isinstance(image,np.ndarray),'input image must be numpy ndarray'
        image=simple_preprocess(image,config.img_size)
        inputs=torch.unsqueeze(torch.from_numpy(image),dim=0).to(self.device)
        outputs=self.model.forward(inputs)
        result=torch.softmax(outputs,dim=1).data.cpu().numpy()
        return np.squeeze(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--app',
                        help='train(train and val) or vis or test(without ground truth)',
                        choices=['train','vis','test'],
                        default='train')

    parser.add_argument('--model_name',
                        help='model name',
                        choices=['vgg11','vgg13'],
                        default='vgg11')

    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['fire','smoking'],
                        default='fire')

    parser.add_argument('--epoch',
                        help='train epoch',
                        type=int,
                        default=30)

    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=4)

    parser.add_argument('--freeze_ratio',
                        help='the freeze ratio for trainning backbone',
                        type=float,
                        default=1.0)

    parser.add_argument('--note',
                        default='one')

    args=parser.parse_args()
    config=edict()
    config.app=args.app
    config.model_name=args.model_name
    config.dataset_name=args.dataset_name
    config.epoch=args.epoch
    config.batch_size=args.batch_size
    assert args.freeze_ratio>=0 and args.freeze_ratio<=1.0
    config.freeze_ratio=args.freeze_ratio
    config.save_model=True
    config.lr=1e-4
    config.img_size=(224,224)
    config.log_dir=os.path.expanduser('~/logs')
    config.root_path=os.path.join('dataset/smoke',config.dataset_name)
    config.note=args.note
    config.load_model_path=os.path.join(config.log_dir,
                                        config.model_name,
                                        config.dataset_name,
                                        config.note)

    t=trainer(config)

    if config.app=='train':
        t.train_val()
    elif config.app=='test':
        input_video_path='/media/sdb/ISCAS_Dataset/driver_videos/抽烟3.h264'
        cap=cv2.VideoCapture(input_video_path)
        assert cap.isOpened(),'cannot open video {}'.format(input_video_path)

        codec = cv2.VideoWriter_fourcc(*"mp4v")
        fps=30
        output_video_path='output.mp4'
        writer=None

        names=['normal','smoking']
        last_img=None
        while True:
            flag,img=cap.read()
            if flag:
                if writer is None:
                    height,width=img.shape[0:2]
                    writer=cv2.VideoWriter(output_video_path,codec,fps,(width,height))

                result=t.test(img)
#                print(names[np.argmax(result)])
                if result[1]>0.9:
                    text='smoking'
                    color=(0,255,0)
                else:
                    text='normal'
                    color=(255,0,0)

                print(text)

                fontScale=max(1,img.shape[1]//448)
                thickness=max(1,img.shape[1]//112)
                cv2.putText(img, text , (50,150), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness)
                # img=np.zeros((224,224,3)); cv2.putText(img, text , (50,150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3); cv2.imwrite('output.jpg',img) ;

                writer.write(img)
                last_img=img.copy()
            else:
                print('end of video')
                break
        writer.release()

        cv2.imwrite('output.jpg',last_img)
    else:
        assert False,'unknown app name {}'.format(config.app)
