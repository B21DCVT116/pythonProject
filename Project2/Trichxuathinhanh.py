import cv2
import os

def extract_face(image_path):
    image = cv2.imread(image_path)

    # Chuyen xang anh xam
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load bộ phân loại khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Nếu không tìm thấy khuôn mặt, trả về None
    if len(faces) == 0:
        print("Không tìm thấy khuôn mặt trong ảnh.")
        return None

    # Lấy khuôn mặt đầu tiên trong danh sách các khuôn mặt được phát hiện
    (x, y, w, h) = faces[0]

    # Trích xuất khuôn mặt từ ảnh gốc
    face = gray[y:y + h, x:x + w]

    # Resize khuôn mặt về kích thước cố định
    face_resized = cv2.resize(face, (500, 500))

    image_with_rect = image.copy()
    cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Anh goc ', image_with_rect)

    cv2.imshow('Mat xam', face_resized)

    # Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Trả về ảnh khuôn mặt đã được resize
    return face_resized


# Đường dẫn đến ảnh


# Gọi hàm để trích xuất và hiển thị khuôn mặt
extract_face(r'D:\Code\PythonProject\Project2\casimytam3.jpg')