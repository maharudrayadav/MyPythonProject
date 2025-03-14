import os
import cv2
import numpy as np
import paramiko
import logging
from flask import request, jsonify
from PIL import Image
from io import BytesIO

# ✅ Load environment variables
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_PATH = "/model/Rudra/lbph_model_{username}.xml"

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_model_from_sftp(username):
    """Loads the LBPH model from SFTP for the given user."""
    model_remote_path = SFTP_REMOTE_PATH.format(username=username)
    transport, sftp = None, None

    try:
        logging.info(f"🔄 Connecting to SFTP: {SFTP_HOST} as {SFTP_USERNAME}...")
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # ✅ Check if file exists
        try:
            file_size = sftp.stat(model_remote_path).st_size
            if file_size == 0:
                logging.error("⚠️ Model file is empty on SFTP")
                return None
        except FileNotFoundError:
            logging.error(f"❌ Model file not found for {username}")
            return None

        # ✅ Read the model file
        with sftp.open(model_remote_path, 'rb') as remote_file:
            model_data = remote_file.read()

        logging.info(f"✅ Model loaded successfully for {username}")
        return BytesIO(model_data)

    except Exception as e:
        logging.error(f"⚠️ SFTP Load Error: {str(e)}")
        return None
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()

def recognize_face(username, image_file):
    """Recognizes a face using the LBPH model stored on SFTP."""
    try:
        # ✅ Load LBPH model from SFTP
        model_data = load_model_from_sftp(username)
        if not model_data:
            return {"error": f"Face model not found for {username}"}

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_data)

        # ✅ Convert image to OpenCV format
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)
        img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # ✅ Detect faces
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return {"message": "No face detected"}

        recognized_faces = []

        # ✅ Compare detected faces with LBPH model
        for (x, y, w, h) in faces:
            face_crop = img_gray[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (100, 100))

            try:
                label, confidence = recognizer.predict(face_crop)
                if confidence < 50:  # Lower confidence means better match
                    recognized_faces.append({"name": username, "confidence": round(100 - confidence, 2)})
                    return {"recognized_faces": recognized_faces}  # ✅ Immediate response if recognized

            except Exception as e:
                return {"error": f"Prediction error: {str(e)}"}

        return {"message": "Face not recognized"}

    except Exception as e:
        logging.error(f"❌ Recognition Error: {str(e)}")
        return {"error": str(e)}
