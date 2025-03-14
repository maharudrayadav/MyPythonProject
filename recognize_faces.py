from flask import request, jsonify
import os
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import paramiko

# Load environment variables
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")

# Face++ API Credentials
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")

# Face++ API URLs
FACE_DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
FACE_COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"

# Dataset Path
DATASET_PATH = "/dataset/Ansh/dataset/Ansh/"
IMAGE_COUNT = 10

# Store Face++ face tokens for stored images
stored_faces = {}

def get_face_token(image_data):
    """Uploads an image to Face++ and gets a face_token."""
    response = requests.post(
        FACE_DETECT_URL,
        data={"api_key": FACEPP_API_KEY, "api_secret": FACEPP_API_SECRET},
        files={"image_file": image_data},
    ).json()

    if "faces" in response and response["faces"]:
        return response["faces"][0]["face_token"]
    return None

def store_face_tokens():
    """Uploads stored images and saves their face tokens."""
    global stored_faces
    stored_faces = {
        f"Ansh_{i}.jpg": get_face_token(open(os.path.join(DATASET_PATH, f"Ansh_{i}.jpg"), "rb"))
        for i in range(1, IMAGE_COUNT + 1)
    }
    print("✅ Stored Face Tokens:", stored_faces)

def load_model_from_sftp(username):
    """Loads the LBPH model from SFTP for the given user."""
    model_remote_path = f"/model/{username}/lbph_model_{username}.xml"
    transport, sftp = None, None

    try:
        print(f"🔄 Connecting to SFTP: {SFTP_HOST} as {SFTP_USERNAME}...")
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Check if file exists and has content
        try:
            file_size = sftp.stat(model_remote_path).st_size
            if file_size == 0:
                print("⚠️ Model file is empty on SFTP")
                return None
        except FileNotFoundError:
            print(f"❌ Model file not found for {username}")
            return None

        with sftp.open(model_remote_path, 'rb') as remote_file:
            model_data = remote_file.read()

        print("✅ Model loaded successfully from SFTP")
        return BytesIO(model_data)
    except Exception as e:
        print(f"⚠️ Error loading model from SFTP: {e}")
        return None
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()

def recognize_with_lbph(username, image):
    """Recognizes a face using the LBPH model."""
    model_data = load_model_from_sftp(username)
    if not model_data:
        return None  # If no model, fallback to Face++

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_stream = BytesIO(model_data)
    recognizer.read(model_stream)

    img_np = np.array(Image.open(image))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (100, 100))

    label, confidence = recognizer.predict(img_gray)
    print(f"🧐 Recognized Label: {label}, Confidence: {confidence}")

    if confidence < 50:
        return {"recognized_faces": [{"name": username, "confidence": round(100 - confidence, 2)}]}

    return None  # If LBPH fails, fallback to Face++

def compare_with_stored_faces(new_image):
    """Compares a new image with stored images using Face++."""
    new_face_token = get_face_token(new_image)
    if not new_face_token:
        return {"message": "No face detected in the new image"}

    best_match = None
    highest_confidence = 0

    for img_name, stored_token in stored_faces.items():
        response = requests.post(
            FACE_COMPARE_URL,
            data={
                "api_key": FACEPP_API_KEY,
                "api_secret": FACEPP_API_SECRET,
                "face_token1": new_face_token,
                "face_token2": stored_token,
            },
        ).json()

        confidence = response.get("confidence", 0)
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_match = img_name

    return {"best_match": best_match, "confidence": highest_confidence}

def recognize_face():
    """Receives image from frontend, compares with stored faces, and returns result."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image received"}), 400

        image_file = request.files["image"]
        username = request.form.get("username", "Unknown")

        # Save temporarily
        temp_path = "/tmp/live_capture.jpg"
        image = Image.open(image_file)
        image.save(temp_path)

        # Try LBPH first
        lbph_result = recognize_with_lbph(username, temp_path)
        if lbph_result:
            return jsonify(lbph_result)

        # Fallback to Face++ if LBPH fails
        result = compare_with_stored_faces(open(temp_path, "rb"))

        return jsonify({"username": username, "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

