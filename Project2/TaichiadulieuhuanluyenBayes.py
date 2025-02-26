from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from Taidulieututhumuc import load_dataset
from Trichxuathinhanh import extract_face

# Tải dữ liệu
dataset_path = r'D:\Code\PythonProject\Project2\PicBTL'
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