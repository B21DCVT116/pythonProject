import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from Trichxuathinhanh import extract_face


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