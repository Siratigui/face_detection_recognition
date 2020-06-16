from django.db import models

# Create your models here.

class TestModel(models.Model):
    personId = models.IntegerField(default=0)
    name = models.CharField(max_length=40, null=True)
    surname = models.CharField(max_length=40, null=True)
    
    
class LastId(models.Model):
    personId = models.IntegerField(default=0)
    