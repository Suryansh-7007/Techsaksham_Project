import cv2
import numpy as np
import sys

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_model.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = {0: "Unknown"}  

cap = cv2.VideoCapture(0)
recognized_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)
        if confidence < 100:  
            name = labels.get(label, "Unknown")
            recognized_names.append(name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(",".join(set(recognized_names)))  