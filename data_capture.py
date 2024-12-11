import cv2
import os
import sys

name = sys.argv[1] 

dataset_path = "dataset"
person_path = os.path.join(dataset_path, name)

if not os.path.exists(person_path):
    os.makedirs(person_path)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
print(f"Capturing faces for {name}. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        count += 1
        file_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(file_path, face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {name}.")
