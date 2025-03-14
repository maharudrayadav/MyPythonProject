import os
import cv2
import numpy as np
import paramiko
import logging
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = 22
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_DATASET_PATH = "/dataset/"
SFTP_REMOTE_MODEL_PATH = "/model/"

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sftp_connect():
    """Establish SFTP connection and return client."""
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return sftp
    except Exception as e:
        logging.error(f"❌ SFTP Connection Error: {str(e)}")
        return None

def ensure_remote_dir(sftp, remote_path):
    """Ensure that the remote directory exists, create it if missing."""
    dirs = remote_path.strip("/").split("/")
    path = "/"
    
    for dir in dirs:
        path = os.path.join(path, dir)
        try:
            sftp.stat(path)
        except FileNotFoundError:
            logging.info(f"📂 Creating missing directory: {path}")
            sftp.mkdir(path)

def download_from_sftp(user_name, local_dataset_path):
    """Download user images from SFTP to local storage."""
    sftp = sftp_connect()
    if not sftp:
        return False

    user_sftp_path = f"{SFTP_REMOTE_DATASET_PATH}{user_name}/"

    try:
        files = sftp.listdir(user_sftp_path)
        logging.info(f"📂 Found {len(files)} files in {user_sftp_path}")
    except FileNotFoundError:
        logging.error(f"❌ Dataset not found: {user_sftp_path}")
        sftp.close()
        return False

    os.makedirs(local_dataset_path, exist_ok=True)

    for file in files:
        remote_file_path = f"{user_sftp_path}{file}"
        local_file_path = os.path.join(local_dataset_path, file)
        try:
            sftp.get(remote_file_path, local_file_path)
            logging.info(f"✅ Downloaded: {file}")
        except Exception as e:
            logging.warning(f"⚠️ Skipped {file}: {str(e)}")

    sftp.close()
    return True

def preprocess_image(image_path):
    """Preprocess image: grayscale, resize, enhance contrast, and crop face."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]

    return img

def augment_image(img):
    """Augment image with flips and rotations."""
    flipped = cv2.flip(img, 1)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return [img, flipped, rotated]

def upload_to_sftp(local_file_path, remote_file_path):
    """Upload trained model to SFTP server."""
    sftp = sftp_connect()
    if not sftp:
        return False

    try:
        ensure_remote_dir(sftp, os.path.dirname(remote_file_path))
        sftp.put(local_file_path, remote_file_path)
        logging.info(f"✅ Model uploaded to {remote_file_path}")
        sftp.close()
        return True
    except Exception as e:
        logging.error(f"❌ SFTP Upload Error: {str(e)}")
        return False

def train_model(user_name):
    """Train LBPH model and upload it to SFTP."""
    local_dataset_path = f"dataset/{user_name}/"

    if not download_from_sftp(user_name, local_dataset_path):
        return {"status": "error", "message": f"Dataset not found for {user_name}"}

    local_model_filename = f"lbph_model_{user_name}.xml"
    remote_model_path = f"{SFTP_REMOTE_MODEL_PATH}{user_name}/{local_model_filename}"

    LBPH_model = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=80)
    images, labels = [], []

    for image_name in os.listdir(local_dataset_path):
        image_path = os.path.join(local_dataset_path, image_name)
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            logging.warning(f"⚠️ Skipping invalid image: {image_name}")
            continue

        for img in augment_image(processed_img):
            images.append(img)
            labels.append(0)  # Single user training

    if not images:
        logging.error(f"❌ No valid images found for {user_name}.")
        return {"status": "error", "message": "No valid images for training."}

    LBPH_model.train(images, np.array(labels))
    LBPH_model.save(local_model_filename)
    logging.info(f"✅ Model trained and saved as {local_model_filename}")

    if upload_to_sftp(local_model_filename, remote_model_path):
        return {"status": "success", "model_path": remote_model_path}
    else:
        return {"status": "error", "message": "Model training completed but upload failed."}
