import numpy as np
import cv2

# Models to detect faces in frames
# Sources:
# - https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
# - https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel

model_address = '../models/deploy.protxt'
weight_address = '../models/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(model_address, weight_address)

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	(h, w) = frame.shape[:2]

	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.9:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
