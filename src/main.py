import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Models to detect faces in frames
# Sources:
# - https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
# - https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel

faceModel = '../models/deploy.protxt'
faceWeights = '../models/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(faceModel, faceWeights)

classify_model = load_model("../models/model_cnn")
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
        if confidence < 0.6:
                continue

        # Finding the box that was detected
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #(startX, startY) = (max(0, startX), max(0, startY))
        #(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, without_mask) = classify_model.predict(face)[0]
        label = "Mask" if mask > without_mask else "No mask" 

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
