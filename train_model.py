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
SFTP_REMOTE_DATASET_PATH = "/dataset/"  # ✅ Ensure leading slash for absolute path
SFTP_REMOTE_MODEL_PATH = "/model/"  # ✅ Path where trained models will be uploaded

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sftp_connect():
    """Establishes an SFTP connection and returns the SFTP client."""
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return sftp
    except Exception as e:
        logging.error(f"❌ SFTP Connection Error: {str(e)}")
        return None

def download_from_sftp(user_name, local_dataset_path):
    """Downloads user images from SFTP to local storage."""
    sftp = sftp_connect()
    if not sftp:
        return False  # Connection failed

    # ✅ Fix path: Avoid double dataset folder
    user_sftp_path = f"{SFTP_REMOTE_DATASET_PATH}{user_name}/dataset/{user_name}/"

    try:
        files = sftp.listdir(user_sftp_path)  # ✅ Check files before accessing the directory
        logging.info(f"📂 Found {len(files)} files in {user_sftp_path}")
    except FileNotFoundError:
        logging.error(f"❌ Error: User dataset not found at {user_sftp_path}")
        sftp.close()
        return False

    if not os.path.exists(local_dataset_path):
        os.makedirs(local_dataset_path)

    for file in files:
        remote_file_path = f"{user_sftp_path}{file}"
        local_file_path = os.path.join(local_dataset_path, file)
        sftp.get(remote_file_path, local_file_path)
        logging.info(f"✅ Downloaded: {file}")

    sftp.close()
    logging.info(f"✅ Successfully downloaded dataset for {user_name} to {local_dataset_path}")
    return True

def upload_to_sftp(local_file_path, remote_file_path):
    """Uploads the trained model to the SFTP server."""
    sftp = sftp_connect()
    if not sftp:
        return False  # Connection failed

    try:
        # ✅ Ensure the remote model directory exists
        remote_dir = os.path.dirname(remote_file_path)
        try:
            sftp.chdir(remote_dir)
        except IOError:
            logging.info(f"📂 Creating remote directory: {remote_dir}")
            sftp.mkdir(remote_dir)

        # ✅ Upload the model file
        sftp.put(local_file_path, remote_file_path)
        logging.info(f"✅ Model uploaded to {remote_file_path}")
        sftp.close()
        return True
    except Exception as e:
        logging.error(f"❌ SFTP Upload Error: {str(e)}")
        return False

def train_model(user_name):
    """Trains the LBPH model for a specific user and uploads it to SFTP."""
    local_dataset_path = f"dataset/{user_name}/dataset/{user_name}"  # ✅ Ensure correct path

    # ✅ Step 1: Download dataset from SFTP
    if not download_from_sftp(user_name, local_dataset_path):
        return {"status": "error", "message": f"Dataset not found for user {user_name}"}

    # ✅ Define model paths
    local_model_filename = f"lbph_model_{user_name}.xml"
    remote_model_path = f"{SFTP_REMOTE_MODEL_PATH}{user_name}/{local_model_filename}"  # ✅ Upload here

    # ✅ Step 2: Train model
    LBPH_model = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []

    for image_name in os.listdir(local_dataset_path):
        image_path = os.path.join(local_dataset_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logging.warning(f"⚠️ Skipping invalid image: {image_name}")
            continue
        
        images.append(img)
        labels.append(0)  # Since it's for one user, use label 0

    if not images:
        logging.error(f"❌ Error: No valid images found for user {user_name}.")
        return {"status": "error", "message": "No valid images found for training."}

    LBPH_model.train(images, np.array(labels))
    LBPH_model.save(local_model_filename)
    logging.info(f"✅ Model trained and saved locally as {local_model_filename}")

    # ✅ Step 3: Upload model to SFTP
    if upload_to_sftp(local_model_filename, remote_model_path):
        return {"status": "success", "model_path": remote_model_path}
    else:
        return {"status": "error", "message": "Model training completed but upload failed."}
