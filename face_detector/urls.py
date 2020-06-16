# -*- coding: utf-8 -*-

from django.urls import path

from . import views
from .controllers import app_controller, person_controller

urlpatterns = [
    path('', views.index, name='index'),
    path('input_form', app_controller.input_form, name='app_controller.input_form'),
    path('save_form', app_controller.save_form, name='app_controller.save_form'),
    path('take_picture', app_controller.take_picture, name='app_controller.take_picture'),
    path('webcam', app_controller.webcam, name='app_controller.webcam'),
    path('train', app_controller.train, name='app_controller.train'),
    
    path('detect', app_controller.detect, name='app_controller.detect'),
    path('get_info', app_controller.get_info, name='app_controller.get_info'),
    path('get_id', app_controller.get_id, name='app_controller.get_id'),
    
    path('person', person_controller.index, name='person_controller.index'),
    path('person/twitter', person_controller.twitter, name='person_controller.twitter'),
    
    path('save_dataset', app_controller.save_dataset, name='app_controller.save_dataset'),
    path('create_dataset', app_controller.create_dataset, name='app_controller.create_dataset')
    
]