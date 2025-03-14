from train_model import train_model
from flask_cors import CORS
from recognize_faces import recognize_face
import os
import cv2
import logging
import paramiko
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# SFTP Configuration
SFTP_HOST = "your-sftp-host"
SFTP_PORT = 22
SFTP_USERNAME = "your-username"
SFTP_PASSWORD = "your-password"
SFTP_MODEL_DIR = "/model/"
LOCAL_MODEL_DIR = "models/"

# Ensure local model directory exists
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Load MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def download_model(username):
    """Download LBPH model from SFTP server."""
    try:
        model_filename = f"lbph_model_{username}.xml"
        remote_path = os.path.join(SFTP_MODEL_DIR, username, model_filename)
        local_path = os.path.join(LOCAL_MODEL_DIR, model_filename)

        # Establish SFTP connection
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        if not os.path.exists(local_path):
            sftp.get(remote_path, local_path)
            logging.info(f"✅ Model downloaded: {remote_path} -> {local_path}")
        else:
            logging.info("🔄 Model already exists locally.")
        
        sftp.close()
        transport.close()
        return local_path
    except Exception as e:
        logging.error(f"❌ Error downloading model: {e}")
        return None

def recognize_face(username, image_file):
    """Recognizes a face using LBPH."""
    model_path = download_model(username)
    if not model_path:
        return {"error": "Face model not found"}, 404

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    image_np = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results.detections:
        logging.error("❌ No faces detected.")
        return {"message": "No faces detected"}

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y, width, height = (
            int(bbox.xmin * w),
            int(bbox.ymin * h),
            int(bbox.width * w),
            int(bbox.height * h),
        )

        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_crop = face_gray[y:y+height, x:x+width]
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue
        face_resized = cv2.resize(face_crop, (200, 200))
        label, confidence = recognizer.predict(face_resized)
        
        return {"recognized_label": label, "confidence": confidence}
    
    return {"message": "Face detected but not recognized"}

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        logging.error("❌ No image uploaded.")
        return jsonify({"error": "No image uploaded"}), 400

    username = request.form.get("username", "").strip()
    if not username:
        logging.error("❌ Username is missing in request.")
        return jsonify({"error": "Username is required"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    logging.info(f"📸 Recognizing face for user: {username}")
    response = recognize_face(username, file)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

