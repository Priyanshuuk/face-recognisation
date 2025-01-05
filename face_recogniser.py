import cv2 as cv
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize variables
cap = cv.VideoCapture(0)  # Open the webcam (0 is the default camera)
faces = cv.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load Haar Cascade
recognizer = cv.face.LBPHFaceRecognizer_create()  # Local Binary Pattern Histogram Recognizer              '''ye wale line nahi aate'''

# Ensure the directory exists to store face data
if not os.path.exists('faces'):
    os.makedirs('faces')

# Capture face data
def capture_faces(name, num_samples=20):
    print(f"Capturing {num_samples} samples for {name}...")

    # Create directory for the person
    if not os.path.exists(f'faces/{name}'):
        os.makedirs(f'faces/{name}')

    samples_collected = 0
    data = []
    while samples_collected < num_samples:
        isTrue, frame = cap.read()
        if not isTrue:
            print("Failed to grab frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces_detected = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            crop_img = frame[y:y + h, x:x + w]
            samples_collected += 1
            cv.imwrite(f'faces/{name}/{samples_collected}.jpg', crop_img)
            data.append((name, crop_img))  # Store the name and the image

            # Draw a rectangle around the face
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow("Face Capture", frame)
        cv.waitKey(1)

    return data







#understand till here




# Train the recognizer using captured data
def train_recognizer(data):
    faces_data = []
    labels = []
    name_labels = {}  # Dictionary to map names to numeric labels
    current_label = 0  # Start labeling from 0

    for name, face in data:
        if name not in name_labels:
            name_labels[name] = current_label
            current_label += 1
        faces_data.append(face)
        labels.append(name_labels[name])

    # Convert all face images to grayscale before training
    faces_data = [cv.cvtColor(face, cv.COLOR_BGR2GRAY) for face in faces_data]

    # Train the recognizer with face images and numeric labels
    recognizer.train(faces_data, np.array(labels))
    recognizer.save('trainer.yml')  # Save the trained model
    # Save name_labels dictionary for later use in recognition
    np.save('name_labels.npy', name_labels)
    print("Recognizer trained and model saved as 'trainer.yml'.")

# Recognize faces and mark attendance
def mark_attendance(person_name):
    # Record attendance
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    attendance.loc[attendance['Name'] == person_name, 'Attendance'] = date_time
    print(f"Attendance marked for {person_name} at {date_time}")

# Start real-time face recognition for attendance
def recognize_faces():
    print("Starting face recognition...")

    # Load the trained model if it exists
    if not os.path.exists('trainer.yml'):
        print("Model not trained yet! Please train the model first.")
        return

    recognizer.read('trainer.yml')  # Load the trained model

    # Load name_labels from the saved file
    if not os.path.exists('name_labels.npy'):
        print("Error: name_labels not found! Please train the model first.")
        return

    name_labels = np.load('name_labels.npy', allow_pickle=True).item()

    attendance_records = []
    recognized_people = set()  # To track people who've already been marked for attendance

    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            print("Failed to grab frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces_detected = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            crop_img = frame[y:y + h, x:x + w]
            
            # Convert cropped face image to grayscale before passing it to the recognizer
            gray_crop = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)  # Convert to grayscale

            label, confidence = recognizer.predict(gray_crop)

            # If confidence is below a threshold (face recognized)
            if confidence < 100:
                person_name = list(name_labels.keys())[list(name_labels.values()).index(label)]  # Get the name from the label
                
                # Check if the person has already had their attendance marked
                if person_name not in recognized_people:
                    mark_attendance(person_name)
                    recognized_people.add(person_name)  # Add the person to the set to avoid marking again

                # Draw a rectangle around the face with the name
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.putText(frame, person_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv.imshow("Face Recognition", frame)

        k = cv.waitKey(1)
        if k == ord('q'):  # Exit when 'q' is pressed
            break

    return attendance_records

# Export attendance to an Excel sheet
def export_to_excel():
    attendance.to_excel('attendance.xlsx', index=False)
    print("Attendance has been exported to 'attendance.xlsx'")

# Create an empty attendance DataFrame
attendance = pd.DataFrame(columns=["Name", "Attendance"])

# Main flow
def main():
    while True:
        print("\n1. Capture Faces")
        print("2. Train Recognizer")
        print("3. Start Attendance")
        print("4. Export Attendance")
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
                    data.append((person_folder, img))
            train_recognizer(data)
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            export_to_excel()
        elif choice == '5':
            cap.release()
            cv.destroyAllWindows()
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the main program
if __name__ == "__main__":
    main()
