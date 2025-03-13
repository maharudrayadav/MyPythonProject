import os
import paramiko
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# ✅ Load environment variables
load_dotenv()

SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = 22
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_PATH = "dataset"  # ✅ Ensure no trailing slash

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# ✅ Establish SFTP Connection Once
sftp_client = None

def get_sftp_client():
    """Creates and returns a persistent SFTP client."""
    global sftp_client
    if sftp_client is None:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp_client = paramiko.SFTPClient.from_transport(transport)
    return sftp_client

def upload_to_sftp(local_path, remote_filename, user_name):
    """Uploads image to SFTP in a user-specific folder."""
    try:
        sftp = get_sftp_client()
        user_name = user_name.lower()  # ✅ Store all usernames in lowercase
        user_sftp_path = os.path.join(SFTP_REMOTE_PATH, user_name)  # ✅ Corrected path

        # ✅ Ensure user directory exists
        try:
            sftp.chdir(user_sftp_path)
        except IOError:
            sftp.mkdir(user_sftp_path)
            sftp.chdir(user_sftp_path)

        remote_path = os.path.join(user_sftp_path, remote_filename)
        sftp.put(local_path, remote_path)

        logging.info(f"✅ Uploaded {remote_filename} to {user_sftp_path}")
        return {"status": "success", "remote_path": remote_path}

    except Exception as e:
        logging.error(f"❌ SFTP Upload Error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    """Handles image uploads, stores 10 images per user, and uploads to SFTP."""
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing file or name"}), 400

    file = request.files["image"]
    user_name = request.form["name"].strip().lower()  # ✅ Convert username to lowercase

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    dataset_path = "dataset"
    user_folder = os.path.join(dataset_path, user_name)
    os.makedirs(user_folder, exist_ok=True)

    # ✅ Count existing images to ensure only 10 are stored
    existing_images = sorted([f for f in os.listdir(user_folder) if f.endswith(".jpg")])
    
    if len(existing_images) >= 10:
        oldest_image = os.path.join(user_folder, existing_images[0])
        os.remove(oldest_image)  # ✅ Remove the oldest image
        existing_images.pop(0)

    # ✅ Determine new filename
    new_image_index = len(existing_images) + 1
    image_filename = f"{user_name}_{new_image_index}.jpg"
    image_path = os.path.join(user_folder, image_filename)
    
    # ✅ Save Image Locally
    file.save(image_path)

    # ✅ Upload to SFTP
    upload_result = upload_to_sftp(image_path, image_filename, user_name)

    return jsonify({
        "message": f"Image {new_image_index}/10 captured successfully",
        "sftp_result": upload_result
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting server on port {port}") 
    app.run(host="0.0.0.0", port=port, debug=False)
