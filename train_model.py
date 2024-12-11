import cv2
import numpy as np
import os

dataset_path = "dataset"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = []
faces = []
label_dict = {}

current_id = 0
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in label_dict:
                label_dict[label] = current_id
                current_id += 1
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_dict[label])

face_recognizer.train(faces, np.array(labels))
face_recognizer.save("trained_model.yml")
print("Model trained and saved as 'trained_model.yml'")
print("Label mapping:", label_dict)
with open("label_mapping.csv", mode="w") as label_file:
    for name, label in label_dict.items():
        label_file.write(f"{label},{name}\n")

