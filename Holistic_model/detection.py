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
cap = cv2.VideoCapture(0)
#Initiate holistic model 
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

	while cap.isOpened():
		ret, frame = cap.read()

		#recolor feed
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False

		#make detection
		results = holistic.process(image)
		print(results)

		#face, pose, left hand, right hand

		#recolor image back to BGR for rendering (opencv love BGR)
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



			#CONNECTIONS la de noi cac vertex lai
		#draw face landmarks
		#ko dung thi go # de off
		#mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,		#or TESSELATION
		#						  mp_drawing.DrawingSpec(color = (130, 155, 0), thickness = -3, circle_radius = 1),
		#						  mp_drawing.DrawingSpec(color = (130, 155, 0), thickness = 2, circle_radius = 4))

		#pose
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
								  mp_drawing.DrawingSpec(color = (0, 140, 230), thickness = -1, circle_radius = 3),
								  mp_drawing.DrawingSpec(color = (0, 140, 230), thickness = 1, circle_radius = 4))

		#right hand
		mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
								  mp_drawing.DrawingSpec(color = (72, 40, 200), thickness = -1, circle_radius = 3),
								  mp_drawing.DrawingSpec(color = (72, 40, 200), thickness = 2, circle_radius = 4))

		#left hand
		mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
								  mp_drawing.DrawingSpec(color = (200, 40, 72), thickness = -1, circle_radius = 3),
								  mp_drawing.DrawingSpec(color = (200, 40, 72), thickness = 2, circle_radius = 4))


		#export coordinate
		try:
			#extract right_hand landmarks
			right_hand = results.right_hand_landmarks.landmark
			right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
			
			#extract left_hand landmarks
			left_hand = results.left_hand_landmarks.landmark
			left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
			
			#concate rows
			row = right_hand_row + left_hand_row

			#make detection
			X = pd.DataFrame([row])
			body_language_class = model.predict(X)[0]
			body_language_prob = model.predict_proba(X)[0]
			print(body_language_class, body_language_prob)

			#grab ear coords
			coords = tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
												 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
												 [640, 480]).astype(int))

			cv2.rectangle(image, (coords[0], coords[1] + 5), (coords[0] + len(body_language_class) * 23, coords[1] - 30),
						  (73, 38, 187), -1)
			cv2.putText(image, body_language_class, coords,
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 240, 230), 2, cv2.LINE_AA)

			#get status box
			cv2.rectangle(image, (0, 0), (250, 60), (73, 38, 187), -1)
			#display class
			cv2.putText(image, 'CLASS',
						  (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 30, 30), 1, cv2.LINE_AA)
			cv2.putText(image, body_language_class.split(' ')[0],
						(90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 240, 230), 2, cv2.LINE_AA)
			#display probability
			cv2.putText(image, 'PROB',
						  (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 30, 30), 1, cv2.LINE_AA)
			cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
						(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 240, 230), 2, cv2.LINE_AA)


		except:
			pass

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime
		cv2.rectangle(image, (0, 435), (50, 490), (73, 38, 187), -1)
		cv2.putText(image, str(int(fps)), (4, 468), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 240, 230), 2, cv2.LINE_AA)

		cv2.imshow('Raw Webcam Feed', image)


		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
