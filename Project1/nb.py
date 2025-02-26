# # ''''
# # Training Multiple Faces stored on a DataBase:
# # 	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc
# # 	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
# # 	==> for using PIL, install pillow library with "pip install pillow"
# #
# # Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
# #
# # Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18
# #
# # '''
# #
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import os
# #
# # # Path for face image database
# # path = 'dataset'
# #
# # recognizer = cv2.face.LBPHFaceRecognizer_create()
# # detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# #
# # # function to get the images and label data
# # def getImagesAndLabels(path):
# #
# #     imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
# #     faceSamples=[]
# #     ids = []
# #
# #     for imagePath in imagePaths:
# #
# #         PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
# #         img_numpy = np.array(PIL_img,'uint8')
# #
# #         id = int(os.path.split(imagePath)[-1].split(".")[1])
# #         faces = detector.detectMultiScale(img_numpy)
# #
# #         for (x,y,w,h) in faces:
# #             faceSamples.append(img_numpy[y:y+h,x:x+w])
# #             ids.append(id)
# #
# #     return faceSamples,ids
# #
# # print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# # faces,ids = getImagesAndLabels(path)
# # recognizer.train(faces, np.array(ids))
# #
# # # Save the model into trainer/trainer.yml
# # recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
# #
# # # Print the numer of faces trained and end program
# # print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
# import cv2
# import numpy as np
# from PIL import Image
# import os
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # Path for face image database
# path = 'dataset'
#
# # Hàm trích xuất đặc trưng từ ảnh (ví dụ: sử dụng HOG)
# def extract_features(image):
#     # Chuyển ảnh sang grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Resize ảnh về kích thước cố định (ví dụ: 64x128)
#     resized = cv2.resize(gray, (64, 128))
#     # Trích xuất HOG features
#     hog = cv2.HOGDescriptor()
#     features = hog.compute(resized)
#     return features.flatten()
#
# # Hàm để lấy dữ liệu và nhãn
# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     features = []
#     labels = []
#
#     for imagePath in imagePaths:
#         # Đọc ảnh
#         image = cv2.imread(imagePath)
#         # Trích xuất đặc trưng
#         feature = extract_features(image)
#         features.append(feature)
#         # Lấy ID từ tên tệp (giả sử tên tệp có dạng userID.số_thứ_tự.jpg)
#         label = int(os.path.split(imagePath)[-1].split(".")[1])
#         labels.append(label)
#
#     return np.array(features), np.array(labels)
#
# # Lấy dữ liệu và nhãn
# print("[INFO] Đang trích xuất đặc trưng...")
# features, labels = getImagesAndLabels(path)
#
# # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
# # Huấn luyện mô hình Naive Bayes
# print("[INFO] Đang huấn luyện mô hình Naive Bayes...")
# model = GaussianNB()
# model.fit(X_train, y_train)
#
# # Đánh giá mô hình
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"[INFO] Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")
#
# # Lưu mô hình (nếu cần)
# import joblib
# joblib.dump(model, 'naive_bayes_face_recognition_model.pkl')
# print("[INFO] Mô hình đã được lưu vào 'naive_bayes_face_recognition_model.pkl'")
import numpy as np
import cv2
import face_recognition_models
from sklearn.model_selection import train_test_split
import os

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def load_and_encode_faces(image_paths, labels):
    face_encodings = []
    face_labels = []

    for image_path, label in zip(image_paths, labels):
        image = face_recognition_models.load_image_file(image_path)
        face_encoding = face_recognition_models.face_encodings(image)

        if len(face_encoding) > 0:
            face_encodings.append(face_encoding[0])
            face_labels.append(label)

    return np.array(face_encodings), np.array(face_labels)

def main():
    # Load your dataset of images and labels
    dataset_path = "trainer/dataset"
    image_paths = []
    labels = []
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        # Kiểm tra xem đây có phải là thư mục không
        if os.path.isdir(person_folder):
            # Duyệt qua từng ảnh trong thư mục con
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image_paths.append(image_path)
                labels.append(person_name)
    # Encode faces
    X, y = load_and_encode_faces(image_paths, labels)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Train Naive Bayes classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Test the classifier
    predictions = nb.predict(X_test)
    accuracy = np.sum(y_test == predictions) / len(y_test)
    print("Naive Bayes classification accuracy", accuracy)

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition_models.face_locations(frame)
        face_encodings = face_recognition_models.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Predict the person using the trained Naive Bayes classifier
            name = nb.predict([face_encoding])[0]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()