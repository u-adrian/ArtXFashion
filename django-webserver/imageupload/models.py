from django.db import models
from django.contrib import admin
# Create your models here.
import uuid
from django.utils.timezone import now


class testfile(models.Model):
    file_name = models.CharField(max_length=30)
    file_image = models.ImageField(upload_to='images')
    
    #file_file = models.FileField(upload_to='./')

class Transfer(models.Model):
    #token = 'k'
    rand = 'randomseed'
    token = models.CharField(primary_key=True, default=uuid.uuid4, max_length=50, editable=False) #.hex[:30].lower() ##dont add uuid4() braces
    person_image = models.ImageField(upload_to=f'images/{rand}')
    style_image = models.ImageField(upload_to=f'images/{rand}')
    person_image_segmentation = models.ImageField(upload_to='images/', default="images/loading.gif")
    style_transfered = models.ImageField(upload_to='images/', default="images/loading.gif")
    created_date = models.DateTimeField(default=now, editable=False) ###allows clearing later on 
    segment_start_x = models.IntegerField(default=0) ###allows clearing later on 
    segment_start_y = models.IntegerField(default=0) ###allows clearing later on 
    progress = models.IntegerField(default=5)  ###


    
admin.site.register(Transfer)
admin.site.register(testfile)