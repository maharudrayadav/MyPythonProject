import cv2
import os
import sys
import time
import paramiko

# ✅ Get person name from command-line argument
if len(sys.argv) < 2:
    print("❌ Error: Name is required")
    sys.exit(1)

person_name = sys.argv[1]
dataset_path = "dataset"
person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    sys.exit(1)

count = 0
max_images = 10
print(f"📸 Capturing face images for {person_name}...")

captured_images = []

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
            cv2.imwrite(image_path, face_crop)
            captured_images.append(image_path)
            count += 1
            print(f"✅ Image {count} saved: {image_path}")

        if count >= max_images:
            break

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"📁 {count} images saved. Uploading to SFTP...")

# 🔹 SFTP Configuration
SFTP_HOST = "eu-west-1.sftpcloud.io"
SFTP_PORT = 22  # Default SFTP port
SFTP_USERNAME = "e714326d13144486afc9979353b4cdb6"
SFTP_PASSWORD = "t4NFOoIuhUqY8866CrMEeMdlOb7wM42N"
REMOTE_PATH = "dataset"  # Change this to your remote directory

try:
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)

    sftp = paramiko.SFTPClient.from_transport(transport)

    # Ensure the remote directory exists
    try:
        sftp.chdir(REMOTE_PATH)
    except IOError:
        sftp.mkdir(REMOTE_PATH)
        sftp.chdir(REMOTE_PATH)

    person_remote_folder = f"{REMOTE_PATH}/{person_name}"
    try:
        sftp.chdir(person_remote_folder)
    except IOError:
        sftp.mkdir(person_remote_folder)

    for img in captured_images:
        remote_file_path = f"{person_remote_folder}/{os.path.basename(img)}"
        sftp.put(img, remote_file_path)
        print(f"✅ Uploaded: {remote_file_path}")

    sftp.close()
    transport.close()
    print("✅ All images uploaded successfully!")

except Exception as e:
    print(f"❌ SFTP Upload Error: {e}")
