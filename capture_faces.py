import cv2
import os
import sys
import time
import paramiko

def capture_faces_function(person_name):
    dataset_path = "dataset"
    person_folder = os.path.join(dataset_path, person_name)
    os.makedirs(person_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    count = 0
    max_images = 10
    captured_images = []

    while count < max_images:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Error: Could not read frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("❌ No faces detected in this frame")
            continue

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            if face_crop.size > 0:
                image_path = os.path.join(person_folder, f"{count+1}.jpg")
                cv2.imwrite(image_path, face_crop)
                captured_images.append(image_path)
                count += 1
                print(f"✅ Image {count} saved: {image_path}")

            if count >= max_images:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captured {count} images for {person_name}!")

# ✅ Ensure this script doesn't run automatically when imported
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Name is required")
        sys.exit(1)

    person_name = sys.argv[1]
    capture_faces_function(person_name)
