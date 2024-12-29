import cv2
import mediapipe as mp						#mp - thu vien mediapipe
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils		#draw line
mp_holistic = mp.solutions.holistic			#import holistic model (face, hand, pose)

#Realtime webcam
cap = cv2.VideoCapture(0)
#Initiate holistic model 
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
		ret, frame = cap.read()

		#recolor feed
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False

		#make detection
		results = holistic.process(image)



#capture landmarks using opencv & CSV
num_coords = len(results.right_hand_landmarks.landmark) + len(results.left_hand_landmarks.landmark)


landmarks = ['class']
for val in range(1, num_coords + 1):
	landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
with open('coords.csv', mode = 'w', newline = '') as f:
	csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
	csv_writer.writerow(landmarks)
