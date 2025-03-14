import os
import cv2
import paramiko
import numpy as np
import mediapipe as mp
import requests

# Load environment variables for security
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY", "GlEhUJAN9QyNepJn6iQawhKjE2hneExF")  # Store in env
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET", "q077-Bc8OdafI5hzViKgnci7g1oKo8Ta")  # Store in env
LOCAL_MODEL_DIR = "temp_models/"

# Face++ API URLs
FACE_DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
FACE_COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)


def download_model(username: str):
    """Download the LBPH model from SFTP for the given username."""
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


def detect_face_facepp(image):
    """Detects face using Face++ API."""
    response = requests.post(
        FACE_DETECT_URL,
        data={"api_key": FACEPP_API_KEY, "api_secret": FACEPP_API_SECRET},
        files={"image_file": image}
    )
    
    result = response.json()
    
    if "faces" in result and len(result["faces"]) > 0:
        face_token = result["faces"][0]["face_token"]
        return face_token
    else:
        return None


def compare_face_facepp(face_token1, face_token2):
    """Compares two faces using Face++ API."""
    response = requests.post(
        FACE_COMPARE_URL,
        data={
            "api_key": FACEPP_API_KEY,
            "api_secret": FACEPP_API_SECRET,
            "face_token1": face_token1,
            "face_token2": face_token2
        }
    )
    
    return response.json()


def recognize_face(username: str, file):
    """Receives an image, detects the face, and compares it using LBPH and Face++ API."""
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

        x, y = max(0, x), max(0, y)
        width, height = min(w - x, width), min(h - y, height)

        if width == 0 or height == 0:
            continue  # Skip invalid faces

        face_crop = img[y:y + height, x:x + width]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        face_gray = cv2.resize(face_gray, (100, 100))  # Resize for LBPH

        # Predict using LBPH
        label, confidence = recognizer.predict(face_gray)
        print(f"🧐 Recognized Label: {label}, Confidence: {confidence}")  # Debugging

        # Use Face++ for verification
        file.seek(0)  # Reset file pointer
        face_token1 = detect_face_facepp(file)

        if not face_token1:
            return {"error": "Face++ failed to detect a face"}, 400

        # Compare with stored model (Assuming an existing face image for comparison)
        stored_face_token = detect_face_facepp(open(model_path, "rb"))

        if not stored_face_token:
            return {"error": "Face++ failed to detect a stored face"}, 400

        comparison_result = compare_face_facepp(face_token1, stored_face_token)

        # Face++ confidence
        facepp_confidence = comparison_result.get("confidence", 0)

        if confidence < 50 or facepp_confidence > 80:
            recognized_faces.append({"name": username, "LBPH_confidence": round(100 - confidence, 2), "Face++_confidence": facepp_confidence})

    return {"recognized_faces": recognized_faces if recognized_faces else "Face not recognized"}
