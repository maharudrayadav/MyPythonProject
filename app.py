import json
import subprocess
import sys
import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from capture_faces import capture_faces_function  # ✅ Import only required function

# ✅ Set up logging for production
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://cloud-app-dlme.onrender.com", "http://localhost:3000"]}})

@app.route("/capture_faces", methods=["POST"])
def capture_faces_endpoint():
    """Receives a person's name and processes images."""
    data = request.get_json()
    if not data or "name" not in data:
        logging.error("❌ Name is missing in request")
        return jsonify({"error": "Name is required"}), 400

    person_name = data["name"]
    logging.info(f"📸 Capturing started for {person_name}")

    # ✅ Ensure this function does not use cv2.VideoCapture(0)
    try:
        capture_faces_function(person_name)
        return jsonify({"message": f"Capturing started for {person_name}"}), 202
    except Exception as e:
        logging.error(f"❌ Error processing {person_name}: {str(e)}")
        return jsonify({"error": "Failed to process request"}), 500

@app.route("/recognize_faces", methods=["POST"])
def recognize_faces():
    """Runs face recognition and returns results."""
    try:
        result = subprocess.run(
            [sys.executable, "recognize_faces.py"],  # ✅ Use sys.executable
            capture_output=True,
            text=True
        )

        if result.stderr:
            logging.error(f"❌ Subprocess error: {result.stderr}")

        if not result.stdout:
            return jsonify({"error": "No output from subprocess"}), 500

        # ✅ Parse JSON output safely
        try:
            json_output = json.loads(result.stdout.strip())
            return jsonify(json_output)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON output"}), 500

    except Exception as e:
        logging.error(f"❌ Exception in recognize_faces: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/train_model", methods=["POST"])
def train_model():
    """Runs the training script for the model."""
    try:
        result = subprocess.run(
            [sys.executable, "train_model.py"],  # ✅ Use sys.executable
            capture_output=True,
            text=True
        )

        if result.stderr:
            logging.error(f"❌ Training error: {result.stderr}")

        if not result.stdout:
            return jsonify({"error": "No output from training process"}), 500

        return jsonify({"message": "Model training completed successfully"}), 200

    except Exception as e:
        logging.error(f"❌ Exception in train_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Use dynamic port for production
    logging.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)  # ✅ Production mode
