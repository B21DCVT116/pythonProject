import os
import numpy as np
from Name1face_dataset import extract_face

dataset_path = 'casimytam3.jpg'

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

    # Chuyển danh sách thành numpy array
    faces_array = np.array(faces)
    labels_array = np.array(labels)

    # Hiển thị kết quả ra màn hình
    print("Faces array:")
    print(faces_array)
    print("\nLabels array:")
    print(labels_array)
    print("\nLabel dictionary:")
    print(label_dict)

# Gọi hàm để thực thi
load_dataset('casimytam3.jpg')