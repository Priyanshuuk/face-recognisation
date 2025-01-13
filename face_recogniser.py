import cv2 as cv
import os
import numpy as np
import pandas as pd
from datetime import datetime

cap = cv.VideoCapture(0)
faces = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

if not os.path.exists('faces'):
    os.makedirs('faces')

attendance = pd.DataFrame(columns=["Name", "Attendance"])

def save_labels(name_labs):
    np.save('name_labs.npy', name_labs)

def load_labels():
    if os.path.exists('name_labs.npy'):
        name_labs = np.load('name_labs.npy', allow_pickle=True).item()
        return name_labs
    else:
        print("Error: name_labs.npy not found!")
        return None

def capture_faces(name, num_samples=20):
    print(f"Capturing {num_samples} samples for {name}...")

    if not os.path.exists(f'faces/{name}'):
        os.makedirs(f'faces/{name}')

    samples_collected = 0
    data = []
    while samples_collected < num_samples:
        isTrue, frame = cap.read()
        if not isTrue:
            print("Failed to grab frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_detected = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            crop_img = frame[y:y + h, x:x + w]
            samples_collected += 1
            cv.imwrite(f'faces/{name}/{samples_collected}.jpg', crop_img)
            data.append((name, crop_img))

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow("Face Capture", frame)
        cv.waitKey(1)

    return data

def train_recognizer(data):
    faces_data = []
    labels = []
    name_labs = {}
    current_label = 0

    for person_folder in os.listdir('faces'):
        for image_file in os.listdir(f'faces/{person_folder}'):
            img = cv.imread(f'faces/{person_folder}/{image_file}')
            name = person_folder

            if name not in name_labs:
                name_labs[name] = current_label
                current_label += 1

            faces_data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
            labels.append(name_labs[name])

    recognizer.train(faces_data, np.array(labels))
    recognizer.save('trainer.yml')

    save_labels(name_labs)

    print("Recognizer trained and model saved as 'trainer.yml'.")

def mark_attendance(name):
    global attendance
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    if name not in attendance['Name'].values:
        new_row = pd.DataFrame({"Name": [name], "Attendance": [date_time]})
        attendance = pd.concat([attendance, new_row], ignore_index=True)
    else:
        attendance.loc[attendance['Name'] == name, 'Attendance'] = date_time

    print(f"Attendance marked for {name} at {date_time}")
    export_to_excel()

def recognize_faces():
    print("Starting face recognition...")

    if not os.path.exists('trainer.yml'):
        print("Model not trained yet! Please train the model first.")
        return

    recognizer.read('trainer.yml')

    name_labs = load_labels()
    if name_labs is None:
        print("Error loading labels! Please train the model first.")
        return

    recognized_people = set()

    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            print("Failed to grab frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_detected = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            crop_img = frame[y:y + h, x:x + w]

            gray_crop = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)

            label, confidence = recognizer.predict(gray_crop)

            if confidence < 100:
                person_name = list(name_labs.keys())[list(name_labs.values()).index(label)]

                if person_name not in recognized_people:
                    mark_attendance(person_name)
                    recognized_people.add(person_name)

                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.putText(frame, person_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv.imshow("Face Recognition", frame)

        k = cv.waitKey(1)
        if k == ord('q'):
            break

def export_to_excel():
    attendance.to_excel('attendance.xlsx', index=False)
    print("Attendance has been exported to 'attendance.xlsx'")

def main():
    while True:
        print("\n1. Capture Faces")
        print("2. Train Recognizer")
        print("3. Start Attendance")
        
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter name for face capture: ")
            capture_faces(name)
        elif choice == '2':
            print("Training recognizer...")
            data = []
            for person_folder in os.listdir('faces'):
                for image_file in os.listdir(f'faces/{person_folder}'):
                    img = cv.imread(f'faces/{person_folder}/{image_file}')
                    name = person_folder
                    data.append((name, img))
            train_recognizer(data)
        elif choice == '3':
            recognize_faces()
        
        elif choice == '5':
            cap.release()
            cv.destroyAllWindows()
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
