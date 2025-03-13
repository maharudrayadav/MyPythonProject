import cv2
import os
import time
import paramiko
from flask import Flask, request, jsonify

app = Flask(__name__)

SFTP_HOST = "eu-west-1.sftpcloud.io"
SFTP_PORT = 22
SFTP_USERNAME = "e714326d13144486afc9979353b4cdb6"
SFTP_PASSWORD = "t4NFOoIuhUqY8866CrMEeMdlOb7wM42N"
REMOTE_PATH = "dataset"

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    data = request.json
    user_name = data.get("name")

    if not user_name:
        return jsonify({"error": "User name is required"}), 400

    dataset_path = "dataset"
    person_folder = os.path.join(dataset_path, user_name)
    os.makedirs(person_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    if not cap.isOpened():
        return jsonify({"error": "Cannot access webcam"}), 500

    count = 0
    max_images = 10
    captured_images = []

    while count < max_images:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Error: Could not read frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        if len(faces) == 0:
            print("❌ No faces detected in this frame")
            continue

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            if face_crop.size > 0:
                image_path = os.path.join(person_folder, f"{count+1}.jpg")
                cv2.imwrite(image_path, face_crop)

                if os.path.exists(image_path):
                    print(f"✅ Image {count+1} saved: {image_path}")
                    captured_images.append(image_path)
                    count += 1
                else:
                    print(f"❌ Image {count+1} failed to save: {image_path}")

            if count >= max_images:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_images) == 0:
        print("❌ No images captured, skipping SFTP upload")
        return jsonify({"error": "No images captured"}), 500

    print(f"📁 Uploading {len(captured_images)} images to SFTP...")

    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # ✅ Ensure the remote folder exists
        sftp.chdir("dataset")
        person_remote_folder = f"dataset/{user_name}"
        try:
            sftp.chdir(person_remote_folder)
        except IOError:
            print(f"⚠ Creating missing SFTP folder: {person_remote_folder}")
            sftp.mkdir(person_remote_folder)
            sftp.chdir(person_remote_folder)

        for img in captured_images:
            remote_file_path = f"{person_remote_folder}/{os.path.basename(img)}"
            sftp.put(img, remote_file_path)
            print(f"✅ Uploaded: {remote_file_path}")

        sftp.close()
        transport.close()
        return jsonify({"message": "Upload successful", "uploaded_files": captured_images}), 200

    except Exception as e:
        print(f"❌ SFTP Upload Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
