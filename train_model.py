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
SFTP_REMOTE_PATH = "/dataset/"  # ✅ Ensure leading slash for absolute path

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_from_sftp(user_name, local_dataset_path):
    """Downloads user images from SFTP to local storage."""
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # ✅ Fix path: Avoid double dataset folder
        user_sftp_path = f"{SFTP_REMOTE_PATH}{user_name}/dataset/{user_name}/"

        try:
            files = sftp.listdir(user_sftp_path)  # ✅ Check files before accessing the directory
            logging.info(f"📂 Found {len(files)} files in {user_sftp_path}")
        except FileNotFoundError:
            logging.error(f"❌ Error: User dataset not found at {user_sftp_path}")
            sftp.close()
            transport.close()
            return False

        if not os.path.exists(local_dataset_path):
            os.makedirs(local_dataset_path)

        for file in files:
            remote_file_path = f"{user_sftp_path}{file}"
            local_file_path = os.path.join(local_dataset_path, file)
            sftp.get(remote_file_path, local_file_path)
            logging.info(f"✅ Downloaded: {file}")

        sftp.close()
        transport.close()
        logging.info(f"✅ Successfully downloaded dataset for {user_name} to {local_dataset_path}")
        return True

    except Exception as e:
        logging.error(f"❌ SFTP Download Error: {str(e)}")
        return False

def train_model(user_name):
    """Trains the LBPH model for a specific user."""
    local_dataset_path = f"dataset/{user_name}/dataset/{user_name}"  # ✅ Ensure correct path

    # ✅ Step 1: Download dataset from SFTP
    if not download_from_sftp(user_name, local_dataset_path):
        return {"status": "error", "message": f"Dataset not found for user {user_name}"}

    model_filename = f"lbph_model_{user_name}.xml"
    LBPH_model = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []

    for image_name in os.listdir(local_dataset_path):
        image_path = os.path.join(local_dataset_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logging.warning(f"⚠️ Skipping invalid image: {image_name}")
            continue
        
        images.append(img)
        labels.append(0)  # Since it's for one user, use label `0`

    if not images:
        logging.error(f"❌ Error: No valid images found for user {user_name}.")
        return {"status": "error", "message": "No valid images found for training."}

    LBPH_model.train(images, np.array(labels))
    LBPH_model.save(model_filename)
    logging.info(f"✅ Model trained and saved as {model_filename}")

    return {"status": "success", "model_path": model_filename}
