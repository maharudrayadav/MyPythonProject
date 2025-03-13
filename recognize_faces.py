import os
import cv2
import numpy as np
import paramiko
import mediapipe as mp
from fastapi import HTTPException, UploadFile

# SFTP Configuration
SFTP_HOST = "your.sftp.server.com"
SFTP_PORT = 22
SFTP_USERNAME = "your_username"
SFTP_PASSWORD = "your_password"
MODEL_PATH = "/model/{username}/face_embedding_{username}.npy"  # Path on SFTP server
LOCAL_MODEL_DIR = "temp_models/"  # Store models locally

# Load Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def download_model(username: str):
    """Downloads the face embeddings model from SFTP if it exists."""
    model_remote_path = MODEL_PATH.format(username=username)
    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"face_embedding_{username}.npy")

    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)  # Ensure local directory exists

        try:
            sftp.stat(model_remote_path)  # Check if file exists
            sftp.get(model_remote_path, local_model_path)  # Download file
            sftp.close()
            transport.close()
            return local_model_path
        except FileNotFoundError:
            sftp.close()
            transport.close()
            raise HTTPException(status_code=404, detail=f"Face embeddings not found on SFTP for user: {username}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SFTP Connection Error: {str(e)}")

async def recognize_face(username: str, file: UploadFile):
    """Receives an image, detects the face, and compares it with stored embeddings."""
    
    model_path = download_model(username)
    stored_embeddings = np.load(model_path)

    # Read image from frontend
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_detection.process(img_rgb)

    if not results.detections:
        return {"message": "No faces detected"}

    recognized_faces = []
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img.shape
        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
        face_crop = img_rgb[y:y + height, x:x + width]

        face_embedding = np.mean(face_crop, axis=(0, 1))  # Simple feature extraction

        distance = np.linalg.norm(stored_embeddings - face_embedding)
        if distance < 10.0:  # Recognition threshold
            recognized_faces.append({"name": username, "confidence": round(100 - distance, 2)})

    return {"recognized_faces": recognized_faces if recognized_faces else "Face not recognized"}
