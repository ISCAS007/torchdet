from celery import Celery
from celery.task.control import revoke
import time
# Celery configuration
config={}
config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery('test', 
    broker=config['CELERY_BROKER_URL'],
    backend=config['CELERY_RESULT_BACKEND'])
celery.conf.update(config)

@celery.task(bind=True,trial=True)
def long_task(self,a,b):
    for i in range(100):
        print(i)
        time.sleep(1)
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': 100,
                                'status': i})
    return a+b