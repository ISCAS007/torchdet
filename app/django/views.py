from app.models import DetectionResult
from app.algorithm import face_detection_algorithm,process_url
from django.shortcuts import render
from django.http import HttpResponse
from app.celery_task import long_task
from celery.task.control import revoke
import json
import time


def index(request):
    context          = {}
    context['title'] = 'video url test'
    context['task_name'] = 'face_detection'
    context['input_name'] = 'video_url'
    return render(request, 'demo.html',context)

    
def face_detection(request):
    try:
        video_url = request.POST['video_url']
#         app=face_detection_algorithm()
#         app.process(video_url)
        process_url(video_url)
    except Exception as e:
        data={'error_code':1,'result':0,'task_name':'face_detection','error_string':e.__str__()}
        return HttpResponse(json.dumps(data), content_type="application/json")
    
    data={'video_url':video_url,'error_code':0,'result':1,'task_name':'face_detection','error_string':''}
    return HttpResponse(json.dumps(data), content_type="application/json")



def celery_test(request):
    data={}
    task=long_task.apply_async()
    data['id']=task.id
    data['start']=task.status
    revoke(task.id)
    data['finished']=task.status
    return HttpResponse(json.dumps(data),content_type="application/json")

