import cv2
import mediapipe as mp						#mp - thu vien mediapipe
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score	#Accuracy metrics
import pickle
import time

#load model
with open('body_language.pkl', 'rb') as f:
	model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils		#draw line
mp_holistic = mp.solutions.holistic			#import holistic model (face, hand, pose)

pTime = 0
cTime = 0

#Realtime webcam
img = cv2.imread("download.png")
cv2.imshow("Image", img)
cv2.waitKey(0)
height, width, channel  = img.shape

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
	results = holistic.process(img)

	#right hand
	mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
							  mp_drawing.DrawingSpec(color = (72, 40, 200), thickness = -1, circle_radius = 3),
							  mp_drawing.DrawingSpec(color = (72, 40, 200), thickness = 2, circle_radius = 4))

	#left hand
	mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
							  mp_drawing.DrawingSpec(color = (200, 40, 72), thickness = -1, circle_radius = 3),
							  mp_drawing.DrawingSpec(color = (200, 40, 72), thickness = 2, circle_radius = 4))

cv2.imshow("Image", img)
cv2.waitKey(0)