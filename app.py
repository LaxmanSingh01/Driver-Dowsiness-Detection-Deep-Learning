import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model(r'E:\Project for Portfolio\Driver Dowsiness Project\model_Inception.h5')

mixer.init()
alert_sound= mixer.Sound(r'E:\Project for Portfolio\Driver Dowsiness Project\alarm.wav')


st.title("Driver Dowsiness Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
Score=0
while run:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    height,width = frame.shape[0:2]
    faces= face_cascade.detectMultiScale(gray,scaleFactor= 1.3, minNeighbors=3)
    eyes= eye_cascade.detectMultiScale(gray,scaleFactor= 1.2, minNeighbors=1)
    
    cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=2 )
        
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=2 )
        
        # preprocessing steps
        eye= frame[ey:ey+eh,ex:ex+w]
        eye= cv2.resize(eye,(80,80))
        eye= eye/255
        eye= eye.reshape(80,80,3)
        eye= np.expand_dims(eye,axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        
        # if eyes are closed
        if prediction[0][0]>0.20:
            cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score=Score+1
            if(Score>5):
                try:
                    alert_sound.play()
                except:
                    pass
            
        # if eyes are open
        elif prediction[0][1]>0.70:
            cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)      
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score = Score-1
            if (Score<0):
                Score=0
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')