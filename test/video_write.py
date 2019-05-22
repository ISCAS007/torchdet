# -*- coding: utf-8 -*-

import os
import cv2
import glob
import skvideo.io
import time
import imageio

class video_readers():
    def __init__(self,root_path,reader='skvideo'):
        self.root_path=root_path
        self.reader=reader
        assert self.reader in ['skvideo','opencv','imageio']
        files=glob.glob(os.path.join(root_path,'**','*'),recursive=True)
        suffix=('mp4','avi','mov','wmv')
        self.video_files=[f for f in files if f.lower().endswith(suffix)]
        self.count=0
        self.cap=None
        self.frame=0
        self.nframes=0
        print(self.video_files)
    
    def __len__(self):
        return len(self.video_files)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cap is None:
            self.new_video()
            
        returned=False
        while not returned:
            while True:
                if self.reader=='skvideo':
                    origin_img=self.cap.__next__()
                elif self.reader=='opencv':
                    flag,origin_img=self.cap.read()
                    if not flag:
                        break
                elif self.reader=='imageio':
                    origin_img=self.cap.get_next_data()
                else:
                    assert False
                    
                self.frame+=1
                
                if self.frame%10==0:
                    returned=True
                    break
                elif self.frame>=self.nframes:
                    break
            
            if returned:
                return_img=origin_img
                break
            else:
                print('frame={},total frame={},video_path={}'.format(self.frame,
                      self.nframes,
                      self.video_files[self.count]))
                
                if self.reader=='skvideo':
                    self.cap.close()
                elif self.reader=='opencv':
                    self.cap.release()
                elif self.reader=='imageio':
                    self.cap.close()
                else:
                    assert False
                    
                if self.count+1>=len(self.video_files):
                    raise StopIteration
                else:
                    self.count+=1
                    self.new_video()
        
        print(self.frame,self.nframes)
        return self.video_files[self.count],return_img
   
    def new_video(self):
        self.frame = 0
        cv_cap = cv2.VideoCapture(filename=self.video_files[self.count],apiPreference=1900)
        
        fps = cv_cap.get(cv2.CAP_PROP_FPS)
        self.nframes = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('fps is {}, total frame is {}, {}'.format(fps,self.nframes,self.video_files[self.count]))
        
        if self.reader=='skvideo':
            cv_cap.release()
            self.cap=skvideo.io.vreader(self.video_files[self.count])
        elif self.reader=='opencv':
            self.cap=cv_cap
        elif self.reader=='imageio':
            self.cap = imageio.get_reader(self.video_files[self.count],'ffmpeg')
        else:
            assert False
        print('read video okay')
        
if __name__ == '__main__':
    root_dir='/media/sdb/ISCAS_Dataset/Sinopec_drone'
    
    results={}
    for reader in ['opencv','imageio']:
        t=time.time()
        readers=video_readers(root_path=root_dir,reader=reader)
        save_dir='output'
        write_path=None
        writer=None
        for path,frame in readers:
            save_path=path.replace(path[-4:],'.mp4')
            save_path=save_path.replace(root_dir,save_dir)
            readers.nframes=100
            assert save_path!=path
            if write_path != save_path:
                write_path=save_path
                print(write_path)
                print(readers.nframes)
        results[reader]=time.time()-t
        
    for k,v in results.items():
        print(k,v)