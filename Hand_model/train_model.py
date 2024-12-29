import cv2
import mediapipe as mp						#mp - thu vien mediapipe
import csv
import uuid
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

mp_drawing = mp.solutions.drawing_utils		#draw line
mp_holistic = mp.solutions.holistic			#import holistic model (face, hand, pose)


#load pose & face data from csv
df = pd.read_csv('coords_test.csv')

X = df.drop('class', axis = 1)	#features
y = df['class']					#target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)


#train sklearn model
pipelines = {
	'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
	'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
	'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
	'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
fit_models = {}
for algo, pipeline in pipelines.items():
	model = pipeline.fit(X_train, y_train)
	fit_models[algo] = model


#evaluate & serialize model
for algo, model in fit_models.items():
	yhat = model.predict(X_test)
	print(algo, accuracy_score(y_test, yhat))

with open('body_language_test.pkl', 'wb') as f:
	pickle.dump(fit_models['rf'], f)