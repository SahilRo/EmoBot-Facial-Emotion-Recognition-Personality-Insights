import cv2
import os


def capture_images(name, num_images):
    data_path = "C:\\Users\\Sahil\\Downloads\\NEW\\" + name + '/'
    os.makedirs(data_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        cv2.imshow('Capture Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            file_name_path = data_path + str(count + 1) + '.jpg'
            cv2.imwrite(file_name_path, frame)
            print(f"Image {count + 1} captured for {name}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    num_persons = int(input("Enter the number of persons to capture images for: "))

    for i in range(num_persons):
        name = input(f"Enter the name of person {i + 1}: ")
        num_images_per_person = int(input(f"Enter the number of images to capture for {name}: "))
        print(f"Capturing {num_images_per_person} images for {name}. Press 'c' to capture each image.")
        capture_images(name, num_images_per_person)
