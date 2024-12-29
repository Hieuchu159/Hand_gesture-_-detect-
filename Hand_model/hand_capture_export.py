import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import csv
import time

pTime = 0
cTime = 0

mp_drawing = mp.solutions.drawing_utils				#draw line
mp_hands = mp.solutions.hands						#import hand model

class_name = "Y"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

	while cap.isOpened():
		ret, frame = cap.read()

		#recolor feed
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		#set flag
		image.flags.writeable = False

		#Dectection
		results = hands.process(image)
		#print(results)


		#set flag to true
		image.flags.writeable = True

		#recolor image back to BGR for rendering (opencv love BGR)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# Rendering results
		if results.multi_hand_landmarks:
			for num, hand in enumerate(results.multi_hand_landmarks):
				mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
										  mp_drawing.DrawingSpec(color = (72, 40, 200), thickness = -1, circle_radius = 3),
										  mp_drawing.DrawingSpec(color = (72, 40, 200), thickness = 2, circle_radius = 4))


		#export coordinate
		try:
			#extract hand landmarks
			all_hands_landmarks = []
			for num, hand in enumerate(results.multi_hand_landmarks):
				all_hands_landmarks.append(hand.landmark)

			Hands = all_hands_landmarks
			# Lặp qua tất cả các landmarks trong biến "Hands" và thu thập thông tin
			landmarks_data = []
			for hand_landmarks in Hands:
			    for landmark in hand_landmarks:
			        x = landmark.x
			        y = landmark.y
			        z = landmark.z
			        visibility = landmark.visibility
			        landmarks_data.append([x, y, z, visibility])

			# Chuyển danh sách các landmarks thành một danh sách phẳng
			landmarks_row = list(np.array(landmarks_data).flatten())


			#append class name
			landmarks_row.insert(0, class_name)

		except:
			pass

		if cv2.waitKey(1) & 0xFF == ord('e'):
			#export to CSV
			with open('coords_test.csv', mode = 'a', newline = '') as f:
				csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
				csv_writer.writerow(landmarks_row)

			cv2.imwrite(os.path.join('Y', '{}.jpg'.format(uuid.uuid1())), image)

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