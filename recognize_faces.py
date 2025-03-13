import os
import cv2
import numpy as np
import paramiko
import mediapipe as mp

# Load environment variables
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_PATH = "model/{username}/face_embedding_{username}.npy"
LOCAL_MODEL_DIR = "temp_models/"

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def download_model(username: str):
    """Downloads face embeddings from SFTP."""
    model_remote_path = f"/model/{username}/face_embedding_{username}.npy"  # Full path
    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"face_embedding_{username}.npy")

    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

        print(f"🔎 Checking file: {model_remote_path}")  # Debugging

        try:
            sftp.stat(model_remote_path)  # Check if file exists
            sftp.get(model_remote_path, local_model_path)
            print(f"✅ Downloaded: {local_model_path}")  # Debugging
            sftp.close()
            transport.close()
            return local_model_path
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_remote_path}")  # Debugging
            sftp.close()
            transport.close()
            return None

    except Exception as e:
        print(f"⚠️ SFTP Error: {e}")
        return None


def recognize_face(username: str, file):
    """Receives an image, detects the face, and compares it with stored embeddings."""
    model_path = download_model(username)

    if not model_path:
        return {"error": f"Face embeddings not found for {username}"}, 404

    stored_embeddings = np.load(model_path)
    print(f"🔍 Loaded embeddings shape: {stored_embeddings.shape}")  # Debugging

    # Read uploaded image
    contents = file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}, 400

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
        print(f"🧐 Extracted face embedding: {face_embedding}")  # Debugging

        distance = np.linalg.norm(stored_embeddings - face_embedding)
        print(f"📏 Distance calculated: {distance}")  # Debugging

        if distance < 10.0:  # Adjust threshold if needed
            recognized_faces.append({"name": username, "confidence": round(100 - distance, 2)})

    return {"recognized_faces": recognized_faces if recognized_faces else "Face not recognized"}

