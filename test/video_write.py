# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import cv2
import glob
import skvideo.io

class video_readers():
    def __init__(self,root_path):
        self.root_path=root_path
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
                origin_img=self.cap.__next__()
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
                print('frame={},total frame={},video_path'.format(self.frame,
                      self.nframes,
                      self.video_files[self.count]))

                self.cap.close()
                if self.count+1>=len(self.video_files):
                    raise StopIteration
                else:
                    self.count+=1
                    self.new_video()
        
        print(self.frame,self.nframes)
        return self.video_files[self.count],return_img
   
    def new_video(self):
        self.frame = 0
        cv_cap = cv2.VideoCapture(self.video_files[self.count])
        
        fps = cv_cap.get(cv2.CAP_PROP_FPS)
        self.nframes = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('fps is {}, total frame is {}'.format(fps,self.nframes))
        cv_cap.release()
        
        self.cap=skvideo.io.vreader(self.video_files[self.count])
        
if __name__ == '__main__':
    root_dir='/media/tmp/Sinopec_drone'
    readers=video_readers(root_path=root_dir)
    save_dir='output'
    
    write_path=None
    writer=None
    for path,frame in readers:
        save_path=path.replace(path[-4:],'.mp4')
        save_path=save_path.replace(root_dir,save_dir)
        assert save_path!=path
        if write_path != save_path:
            write_path=save_path
            print(write_path)