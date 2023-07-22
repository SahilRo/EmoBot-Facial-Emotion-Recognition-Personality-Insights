import cv2
import numpy as np
import os

def load_dataset(data_path):
    person_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    Training_Data, Labels = [], []
    person_label = 0

    for person_folder in person_folders:
        person_path = os.path.join(data_path, person_folder)
        person_images = [f for f in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, f))]

        for person_image in person_images:
            image_path = os.path.join(person_path, person_image)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(image, dtype=np.uint8))
            Labels.append(person_label)

        person_label += 1

    Labels = np.asarray(Labels, dtype=np.int32)
    return Training_Data, Labels

def train_model(data_path):
    Training_Data, Labels = load_dataset(data_path)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))

    person_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    labels_dict = {i: person_folders[i] for i in range(len(person_folders))}

    return model, labels_dict

def initialize_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe(
        "deploy_age.prototxt",
        "age_net.caffemodel")

    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt",
        "gender_net.caffemodel")

    return (age_net, gender_net)

if __name__ == "__main__":
    data_path = 'C:\\Users\\Sahil\\Downloads\\New\\'
    model, labels_dict = train_model(data_path)
    print("Dataset Model Training Complete!!!!!")
    print(labels_dict)
