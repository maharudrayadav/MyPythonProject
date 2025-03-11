import cv2
import os
import sys
import time

# ✅ Ensure person name is provided
if len(sys.argv) < 2:
    print("❌ Error: Name is required")
    sys.exit(1)

person_name = sys.argv[1]  # Read from command-line argument

# ✅ Use absolute path for the dataset folder
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
dataset_path = os.path.join(script_dir, "dataset")  # Ensure absolute path
person_folder = os.path.join(dataset_path, person_name)

# ✅ Create directories if they don't exist
os.makedirs(person_folder, exist_ok=True)

# ✅ Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open webcam
time.sleep(2)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    sys.exit(1)

count = 0
max_images = 10
print(f"📸 Capturing face images for {person_name}...")

while count < max_images:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Error: Could not read frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_crop = gray[y:y + h, x:x + w]

        if face_crop.size > 0:
            image_path = os.path.join(person_folder, f"{count}.jpg")
            
            # ✅ Debugging: Print image storage path
            print(f"✅ Saving image {count + 1} to: {image_path}")

            cv2.imwrite(image_path, face_crop)
            count += 1

        if count >= max_images:
            break

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"📁 {count} images saved in: {person_folder}")
