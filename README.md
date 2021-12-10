# Driver-Dowsiness-Detection-Deep-Learning

## Dataset Link :  http://mrl.cs.vsb.cz/eyedataset

This code can detect your eyes and alert when the user is drowsy.

## Applications
This can be used by riders who tend to drive for a longer period of time that may lead to accidents

## Dependencies
1 import cv2
2 import streamlit as st
3 import tensorflow as tf
4 from tensorflow.keras.models import load_model
5 import numpy as np
6 from pygame import mixer

## Description
A computer vision system that can automatically detect driver drowsiness in a real-time video stream and then play an alarm if the driver appears to be drowsy.

## Algorithm

Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the eye.

It checks 20 consecutive frames and if the Eye Aspect ratio is less than 0.25, Alert is generated.

![image](https://user-images.githubusercontent.com/84785447/145530429-803bb6e1-399e-4590-b8ef-1aceee45c38f.png)

Summing up:

![image](https://user-images.githubusercontent.com/84785447/145530534-266b30a8-3c04-48aa-94ab-126fdeff1612.png)  ![image](https://user-images.githubusercontent.com/84785447/145530646-1586d3d1-f136-4ef5-9e17-cb44dc05712a.png)

![image](https://user-images.githubusercontent.com/84785447/145530711-12607142-80f6-4ac0-bfa6-db79efc379b2.png)

## Execution 

To run the code, type streamlit run app.py


