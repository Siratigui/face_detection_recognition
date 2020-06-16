# -*- coding: utf-8 -*-

from django.shortcuts import render, redirect

def index(request):
    return render(request, 'person/index.html')



def twitter(request):
    return render(request, 'person/twitter/index.html')