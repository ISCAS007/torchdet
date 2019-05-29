from django.db import models

class DetectionResult(models.Model):
    video_url=models.TextField()
    image_path=models.TextField()
    date=models.DateField()
    bbox=models.TextField()