import cv2
import numpy as np
from fer import FER
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
        "C:\\Users\\Sahil\\Downloads\\deploy_age_prototxt.txt",
        "C:\\Users\\Sahil\\Downloads\\age_net (1).caffemodel")

    gender_net = cv2.dnn.readNetFromCaffe(
        "C:\\Users\\Sahil\\Downloads\\deploy_gender_prototxt.txt",
        "C:\\Users\\Sahil\\Downloads\\gender_net.caffemodel")

    return (age_net, gender_net)

def read_from_camera(cap, age_net, gender_net, model, labels_dict, Training_Data):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, image = cap.read()

        face_cascade = cv2.CascadeClassifier("C:\\Users\\Sahil\\Downloads\\project\\haarcascade_frontalface_default.xml")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

            face_img = image[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender: " + gender)

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            # Predict Emotion
            emo_detector = FER(mtcnn=True)
            captured_emotions = emo_detector.detect_emotions(face_img)
            dominant_emotion, emotion_score = emo_detector.top_emotion(face_img)
            print("Dominant emotion: {}, Score: {}".format(dominant_emotion, emotion_score))

            # Predict Name
            face_region = gray[y:y + h, x:x + w]
            face_encoding = model.predict(face_region)[0]
            match = [np.array_equal(face_encoding, encoding) for encoding in Training_Data]
            if any(match):
                index = np.where(match)[0][0]
                name = labels_dict[Labels[index]]
            else:
                name = "Unknown"

            overlay_text = "{} {} {} Emotion: {}".format(name, gender, age, dominant_emotion)
            cv2.putText(image, overlay_text, (x, y - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    age_net, gender_net = initialize_caffe_models()

    data_path = 'C:\\Users\\Sahil\\Downloads\\New\\'
    model, labels_dict = train_model(data_path)
    print("Dataset Model Training Complete!!!!!")

    Training_Data, _ = load_dataset(data_path)

    cap = cv2.VideoCapture(0)
    read_from_camera(cap, age_net, gender_net, model, labels_dict, Training_Data)