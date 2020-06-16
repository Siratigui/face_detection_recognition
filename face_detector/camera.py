# -*- coding: utf-8 -*-

from django.shortcuts import redirect
import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_detected = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        frame_flip = cv2.flip(image,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()
    
    def generate_dataset(self, img, id, img_id):
        cv2.imwrite("face_detector/data/user."+str(id)+"."+str(img_id)+".jpg", img)

    def get_dataframe(self, img_id):
        success, img = self.video.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        coords = []
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Face", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
            
            coords = [x, y, w, h]
            
        if len(coords) == 4:
            roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            id = 1
            self.generate_dataset(roi_img, id, img_id)
       
        frame_flip = cv2.flip(img, 1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes(), coords
    
        