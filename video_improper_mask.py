from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import math

import os
import sys
from threading import Timer
import shutil
import time

detections = None 
def detect_and_predict_mask(frame, faceNet, maskNet,threshold):
	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence >threshold:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			locs.append((startX, startY, endX, endY))
			preds.append(maskNet.predict(face)[0].tolist())
	return (locs, preds)


MASK_MODEL_PATH=os.getcwd()+"\\model.h5"
FACE_MODEL_PATH=os.getcwd()+"\\face_detector" 
THRESHOLD = 0.5

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(MASK_MODEL_PATH)

print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	original_frame = frame.copy()
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,THRESHOLD)
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(correct,incorrect,without)=pred
		label = "Mask" if correct > incorrect else "No Mask"
		wi=incorrect+without
		cw=correct+without
		ci=correct+incorrect
		if correct > wi:
			label="Mask"
			color = (0, 255, 0)
		elif incorrect> cw:
			label="Incorrect"
			color = (255,0, 0)
		elif without > ci:
			label="No Mask"
			color = (0,0, 255)
		else:
			label="----"
		whitecolor=(255,255, 255)
		label = "{}{:.2f}%".format(label, max(correct,incorrect,without) * 100)
		y0, dy=startX, startY - 15
		for i, line in enumerate(label.split('\n')):
			cv2.putText(original_frame, line, (startX, dy),cv2.cv2.FONT_HERSHEY_DUPLEX, 0.3, whitecolor)
			dy=dy+10
		cv2.rectangle(original_frame, (startX, startY), (endX, endY), color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, startY-23), color, -2)
    

	cv2.addWeighted(frame, 0.5, original_frame, 0.5 , 0,frame)
	frame= cv2.resize(frame,(860,490))
	cv2.imshow("Incorrect-Correct-Without Mask Detection App", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
