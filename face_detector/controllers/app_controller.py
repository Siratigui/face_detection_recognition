# -*- coding: utf-8 -*-

from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse, HttpResponse
from django.http import JsonResponse
from face_detector.controllers.functions import custom_redirect
from face_detector.camera import VideoCamera
import cv2
import numpy as np
import os
from PIL import Image

from face_detector.models import TestModel, LastId

def input_form(request):
    return render(request, 'face_detector/input_form.html')

def save_form(request):
    post_personId = request.POST['id']
    post_name = request.POST['name']
    post_surname = request.POST['surname']
    
    t = TestModel(
            personId = post_personId,
            name = post_name, 
            surname = post_surname
            )
    t.save()
    
    return custom_redirect('app_controller.take_picture', id=post_personId)

def take_picture(request):
    get_id = request.GET['id']
    return render(request, 'face_detector/take_picture.html', {'id': get_id})

def get_info(request):
    LastId.objects.all().delete()
    return render(request, 'face_detector/get_info.html')

def get_id(request):
    id = LastId.objects.latest('personId')
    return HttpResponse(str(id.personId));


def create_dataset(request):
    return render(request, 'face_detector/create_dataset.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
      

        
#def webcam(request):
#    return StreamingHttpResponse(gen(VideoCamera()),
#                                 content_type='multipart/x-mixed-replace; boundary=frame')
    
    

def save_dataset(request):
    
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    
    post_personId = request.POST['personId']
    post_name = request.POST['name']
    post_surname = request.POST['surname']
    sampleNum = 0
    while(True):
        ret, img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            sampleNum = sampleNum+1
            cv2.imwrite('face_detector/data/user.'+str(post_personId)+'.'+str(sampleNum)+'.jpg', img[y:y+h,x:x+w])
            
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Face", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
            cv2.waitKey(250)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if sampleNum>10:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    
    t = TestModel(
            personId = post_personId,
            name = post_name, 
            surname = post_surname
            )
    t.save()
    return redirect('index')



def crazy_test(id):
    
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    sampleNum = 0
    while True:
        ret, img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        
        for(x,y,w,h) in faces:
            cv2.imwrite('face_detector/data/user.'+str(id)+'.'+str(sampleNum)+'.jpg', img[y:y+h,x:x+w])
            sampleNum = sampleNum+1
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Me", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
#        frame_flip = cv2.flip(img, 1)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        
        yield(b'--frame\r\n'
                  b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        if sampleNum>10:
            break
    video_capture.release()
    
        
def webcam(request):
    get_id = request.GET['id']
    return StreamingHttpResponse(crazy_test(get_id),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def train(request):
    data_dir = 'face_detector/data'
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        
        faces.append(imageNp)
        ids.append(id)
        
    ids = np.array(ids)
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.save('face_detector/brains/detection_model.yml')

    return redirect('index')


def detecter():
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read('face_detector/brains/detection_model.yml')
    Gid = 0
    while True:
        ret, img = video_capture.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray_img, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            id,conf = clf.predict(gray_img[y:y+h, x:x+w])
            
            print(conf)
#            if conf<35:
#                userId = getId
#                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
            Gid = id
            obj = TestModel.objects.get(personId = id)
            cv2.putText(img, obj.name, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
            
#            else:
#                cv2.putText(img, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        
        yield(b'--frame\r\n'
                  b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        if Gid != 0:
            break
    video_capture.release()
    l = LastId(personId = Gid)
    l.save()

def detect(request):
#    num = request.session.get('num')
    return StreamingHttpResponse(detecter(), 
                                 content_type='multipart/x-mixed-replace; boundary=frame')



#def ten(camera):
#    sampleNum = 0
#    img_id = 0
#    while True:
#        frame, coords = camera.get_dataframe(img_id)
#        
#        if len(coords) == 4:
#            sampleNum += 1
#        img_id += 1
#        
#        if sampleNum>10:
#            break
#        yield (b'--frame\r\n'
#               b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#    del(camera)



#def save_dataset(request):
#    return StreamingHttpResponse(ten(VideoCamera()),
#                                 content_type='multipart/x-mixed-replace; boundary=frame')





