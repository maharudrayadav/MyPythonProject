import os
import cv2
import paramiko
import numpy as np
import mediapipe as mp

# Load environment variables
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
LOCAL_MODEL_DIR = "temp_models/"

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def download_model(username: str):
    """Download the LBPH model from the SFTP server for a given username."""
    model_remote_path = f"/model/{username}/lbph_model_{username}.xml"
    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"lbph_model_{username}.xml")

    transport, sftp = None, None
    try:
        print(f"🔄 Connecting to SFTP: {SFTP_HOST} as {SFTP_USERNAME}...")
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

        print(f"🔍 Checking if model exists: {model_remote_path}")

        try:
            sftp.stat(model_remote_path)  # Check if file exists
            sftp.get(model_remote_path, local_model_path)
            print(f"✅ Model downloaded successfully: {local_model_path}")
            return local_model_path
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_remote_path}")
            return None
    except paramiko.SSHException as e:
        print(f"⚠️ SSH Connection Error: {e}")
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
    if not contents:
        return {"error": "Empty image file"}, 400

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print("❌ Image decoding failed.")
        return {"error": "Invalid image format"}, 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results.detections:
        return {"message": "No faces detected in the image"}, 400

    recognized_faces = []
    h, w, _ = img.shape

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        width, height = int(bbox.width * w), int(bbox.height * h)

        # Ensure bounding box is within image dimensions
        x, y = max(0, x), max(0, y)
        width, height = min(w - x, width), min(h - y, height)

        if width == 0 or height == 0:
            continue  # Skip invalid faces

        face_crop = img[y:y + height, x:x + width]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        face_gray = cv2.resize(face_gray, (100, 100))  # Resize to match LBPH input size

        # Predict using LBPH
        label, confidence = recognizer.predict(face_gray)
        print(f"🧐 Recognized Label: {label}, Confidence: {confidence}")  # Debugging

        if confidence < 50:  # More flexible threshold
            recognized_faces.append({"name": username, "confidence": round(100 - confidence, 2)})

    return {"recognized_faces": recognized_faces if recognized_faces else "Face not recognized"}
