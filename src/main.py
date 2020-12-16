import numpy as np
import cv2

# Models to detect faces in frames
# Sources:
# - https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
# - https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel

faceModel = '../models/deploy.protxt'
faceWeights = '../models/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(faceModel, faceWeights)

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	(h, w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# Check the detection threshold
		if confidence < 0.9:
			continue

		# Finding the box that was detected
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		#(startX, startY) = (max(0, startX), max(0, startY))
		#(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
