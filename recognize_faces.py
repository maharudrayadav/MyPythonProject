import os
import cv2
import paramiko
import numpy as np
import mediapipe as mp
import requests
from io import BytesIO

# Load environment variables
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")

# Face++ API URLs
FACE_DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
FACE_COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def load_model_from_sftp(username: str):
    """Loads LBPH model from SFTP."""
    model_remote_path = f"/model/{username}/lbph_model_{username}.xml"
    transport, sftp = None, None

    try:
        print(f"🔄 Connecting to SFTP: {SFTP_HOST} as {SFTP_USERNAME}...")
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Check if file exists and has content
        file_size = sftp.stat(model_remote_path).st_size
        if file_size == 0:
            print("⚠️ Model file is empty on SFTP")
            return None

        with sftp.open(model_remote_path, 'rb') as remote_file:
            model_data = remote_file.read()

        print("✅ Model loaded successfully from SFTP")
        return BytesIO(model_data)
    except FileNotFoundError:
        print(f"❌ Model file not found for {username}")
        return None
    except Exception as e:
        print(f"⚠️ Error loading model from SFTP: {e}")
        return None
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()

def recognize_face(username: str, file):
    """Recognizes a face using LBPH and falls back to Face++ API if needed."""
    model_data = load_model_from_sftp(username)
    if not model_data:
        return {"error": f"Face model not found for {username}"}, 404

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_data)

    contents = file.read()
    if not contents:
        return {"error": "Empty image file"}, 400

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image format"}, 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results.detections:
        return {"message": "No faces detected in the image"}, 400

    h, w, _ = img.shape
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        width, height = int(bbox.width * w), int(bbox.height * h)

        x, y = max(0, x), max(0, y)
        width, height = min(w - x, width), min(h - y, height)
        if width == 0 or height == 0:
            continue

        face_crop = img[y:y + height, x:x + width]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        face_gray = cv2.resize(face_gray, (100, 100))

        label, confidence = recognizer.predict(face_gray)
        print(f"🧐 Recognized Label: {label}, Confidence: {confidence}")

        if confidence < 50:
            return {"recognized_faces": [{"name": username, "confidence": round(100 - confidence, 2)}]}

    # Fallback to Face++ if LBPH fails
    print("🔄 Face not recognized by LBPH, using Face++...")

    files = {
        "api_key": (None, FACEPP_API_KEY),
        "api_secret": (None, FACEPP_API_SECRET),
        "image_file": file,
    }
    facepp_response = requests.post(FACE_DETECT_URL, files=files).json()
    return facepp_response

def facepp_compare_images(image1, image2):
    """Compares two images using Face++ API."""
    files = {
        "api_key": (None, FACEPP_API_KEY),
        "api_secret": (None, FACEPP_API_SECRET),
        "image_file1": open(image1, "rb"),
        "image_file2": open(image2, "rb"),
    }
    response = requests.post(FACE_COMPARE_URL, files=files)
    return response.json()
