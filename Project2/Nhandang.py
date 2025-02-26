import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from TaichiadulieuhuanluyenBayes import recognize_face,model,label_dict

new_image_path = 'casisontung2.jpg'
recognized_name = recognize_face(new_image_path, model, label_dict)
print(f'Ket qua: {recognized_name}')


image = cv2.imread(new_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imshow('Ket qua khuon mat', image)
cv2.waitKey(0)
cv2.destroyAllWindows()