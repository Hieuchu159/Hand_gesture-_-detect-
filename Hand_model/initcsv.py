import cv2
import mediapipe as mp						#mp - thu vien mediapipe
import csv
import os
import numpy as np
import uuid


mp_drawing = mp.solutions.drawing_utils				#draw line
mp_hands = mp.solutions.hands						#import hand model

#Realtime webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:\

	ret, frame = cap.read()

	#recolor feed
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#set flag
	image.flags.writeable = False

	#Dectection
	results = hands.process(image)


num_coords = 0

#capture landmarks using opencv & CSVq
for num, hand in enumerate(results.multi_hand_landmarks):
	num_coords = num_coords + len(hand.landmark)


landmarks = ['class']
for val in range(1, num_coords + 1):
	landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
with open('coords_test.csv', mode = 'w', newline = '') as f:
	csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
	csv_writer.writerow(landmarks)
