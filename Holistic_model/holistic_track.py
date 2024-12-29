import cv2
import mediapipe as mp						#mp - thu vien mediapipe
import time

pTime = 0
cTime = 0

mp_drawing = mp.solutions.drawing_utils		#draw line
mp_holistic = mp.solutions.holistic			#import holistic model (face, hand, pose)

#Realtime webcam
cap = cv2.VideoCapture(0)
#Initiate holistic model 
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

	while cap.isOpened():
		ret, frame = cap.read()

		#recolor feed
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#flip screen
		image = cv2.flip(image, 1)
		#make detection
		results = holistic.process(image)
		#print(results._____)

		#face, pose, left hand, right hand

		#recolor image back to BGR for rendering (opencv love BGR)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			#CONNECTIONS la de noi cac vertex lai
		#draw face landmarks
		#ko dung thi go # de off
		mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,		#or TESSELATION
								  mp_drawing.DrawingSpec(color = (130, 155, 0), thickness = -3, circle_radius = 1),
								  mp_drawing.DrawingSpec(color = (130, 155, 0), thickness = 2, circle_radius = 4))

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