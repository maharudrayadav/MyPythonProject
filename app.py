import os
import paramiko
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Secure SFTP Configuration
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = 22  # Default SFTP Port
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_PATH = "dataset/"

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

def upload_to_sftp(local_path, remote_filename):
    """Uploads an image to the SFTP server securely."""
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # ✅ Ensure remote directory exists
        try:
            sftp.chdir(SFTP_REMOTE_PATH)
        except IOError:
            sftp.mkdir(SFTP_REMOTE_PATH)
            sftp.chdir(SFTP_REMOTE_PATH)

        remote_path = f"{SFTP_REMOTE_PATH}/{remote_filename}"
        sftp.put(local_path, remote_path)
        sftp.close()
        transport.close()

        logging.info(f"✅ Uploaded {remote_filename} to {SFTP_REMOTE_PATH}")
        return {"status": "success", "remote_path": remote_path}

    except Exception as e:
        logging.error(f"❌ SFTP Upload Error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    """Handles image upload and sends it to the SFTP server."""
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing file or name"}), 400

    file = request.files["image"]
    user_name = request.form["name"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # ✅ Save image locally
    dataset_path = "dataset"
    person_folder = os.path.join(dataset_path, user_name)
    os.makedirs(person_folder, exist_ok=True)

    image_path = os.path.join(person_folder, file.filename)
    file.save(image_path)

    # ✅ Upload to SFTP
    upload_result = upload_to_sftp(image_path, file.filename)

    return jsonify({"message": "Image captured successfully", "sftp_result": upload_result}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
