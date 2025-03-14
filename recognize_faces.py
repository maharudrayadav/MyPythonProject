import os
import cv2
import paramiko
import numpy as np
import mediapipe as mp

# Load environment variables
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_PATH = "model/{username}/lbph_model_{username}.xml"
LOCAL_MODEL_DIR = "temp_models/"

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def download_model(username: str):
    model_remote_path = f"/model/{username}/lbph_model_{username}.xml"
    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"lbph_model_{username}.xml")

    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

        print(f"🔎 Checking file: {model_remote_path}")

        try:
            sftp.stat(model_remote_path)  # Check if file exists
            sftp.get(model_remote_path, local_model_path)
            print(f"✅ Downloaded: {local_model_path}")
            return local_model_path
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_remote_path}")
            return None
    except Exception as e:
        print(f"⚠️ SFTP Error: {e}")
        return None
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


def recognize_face(username: str, file):
    """Receives an image, detects the face, and compares it with the stored LBPH model."""
    model_path = download_model(username)

    if not model_path:
        return {"error": f"Face model not found for {username}"}, 404

    # Load LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

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
        face_crop = img[y:y + height, x:x + width]

        # Convert to grayscale for LBPH
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

        # Predict using LBPH
        label, confidence = recognizer.predict(face_gray)
        print(f"🧐 Recognized Label: {label}, Confidence: {confidence}")  # Debugging

        if confidence < 70:  # Lower confidence means better match
            recognized_faces.append({"name": username, "confidence": round(100 - confidence, 2)})

    return {"recognized_faces": recognized_faces if recognized_faces else "Face not recognized"}
