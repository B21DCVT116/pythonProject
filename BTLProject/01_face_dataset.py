import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Trich xuat hinh anh
def extract_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (100, 100))  # Resize về kích thước cố định
    return face


# Tai du lieu tu thu muc
def load_dataset(dataset_path):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_dict[current_label] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            face = extract_face(image_path)
            if face is not None:
                faces.append(face.flatten())
                labels.append(current_label)

        current_label += 1

    return np.array(faces), np.array(labels), label_dict


# Tải dữ liệu
dataset_path = 'PicBTL'
faces, labels, label_dict = load_dataset(dataset_path)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Huan luyen Bayes
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Ty le phan tram: {accuracy * 100:.2f}%')


def recognize_face(image_path, model, label_dict):
    face = extract_face(image_path)
    if face is None:
        return "Khong tim thay"

    face = face.flatten().reshape(1, -1)
    prediction = model.predict(face)
    return label_dict[prediction[0]]


# Nhận dạng khuôn mặt từ hình ảnh mới
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