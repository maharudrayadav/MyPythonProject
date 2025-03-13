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
SFTP_REMOTE_PATH = "dataset/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def upload_to_sftp(local_path, remote_filename, user_name):
    """Uploads file to SFTP in the user's folder."""
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        user_sftp_path = f"{SFTP_REMOTE_PATH}/{user_name}"
        
        # ✅ Ensure user directory exists
        try:
            sftp.chdir(user_sftp_path)
        except IOError:
            sftp.mkdir(user_sftp_path)
            sftp.chdir(user_sftp_path)

        remote_path = f"{user_sftp_path}/{remote_filename}"
        sftp.put(local_path, remote_path)

        sftp.close()
        transport.close()
        logging.info(f"✅ Uploaded {remote_filename} to {user_sftp_path}")
        return {"status": "success", "remote_path": remote_path}

    except Exception as e:
        logging.error(f"❌ SFTP Upload Error: {str(e)}")
        return {"status": "error", "message": str(e)}

def train_model(user_name):
    """Trains the LBPH model for a specific user."""
    dataset_path = f"dataset/{user_name}/dataset/{user_name}"
    model_filename = f"lbph_model_{user_name}.xml"

    if not os.path.exists(dataset_path):
        logging.error(f"❌ Error: User dataset not found at {dataset_path}")
        return {"status": "error", "message": f"Dataset not found for user {user_name}"}

    LBPH_model = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(0)  # Since it's for one user, use label `0`

    if len(images) == 0:
        logging.error(f"❌ Error: No images found for user {user_name}.")
        return {"status": "error", "message": "No images found for training."}

    LBPH_model.train(images, np.array(labels))
    LBPH_model.save(model_filename)
    logging.info(f"✅ Model trained and saved as {model_filename}")

    return {"status": "success", "model_path": model_filename}
