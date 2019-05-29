import cv2
from celery import Celery
from celery import shared_task

class face_detection_algorithm():
    def __init__(self):
        self.cap=None
    
    @shared_task
    def process(self,video_url):
        if self.cap is not None:
            self.cap.release()

        self.cap=cv2.VideoCapture(video_url)
        if self.cap.isOpened():
            for idx in range(100):
                flag,img=self.cap.read()
                if flag:
                    print('process img')
                else:
                    print('read img failed')
        else:
            raise Exception('cannot open video url {}'.format(video_url))
            
@shared_task(name='process_url')
def process_url(video_url):
    print(video_url)
