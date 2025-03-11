import json
from flask import Flask, jsonify, request
import subprocess
import os

app = Flask(__name__)

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    data = request.json
    person_name = data.get("name")
    image_data = data.get("image")  # Base64 encoded image

    if not person_name or not image_data:
        return jsonify({"error": "Name and image are required"}), 400

    # ✅ Send the Base64 image to capture_faces.py
    process = subprocess.Popen(
        ["python", "capture_faces.py", person_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=image_data)

    if stderr:
        return jsonify({"error": stderr.strip()}), 500

    return jsonify({"message": stdout.strip()}), 200

@app.route("/recognize_faces", methods=["POST"])
def recognize_faces():
    try:
        result = subprocess.run(
            ["python", "recognize_faces.py"],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        if result.stderr:
            print("❌ Subprocess error:", result.stderr)

        if not result.stdout:
            return jsonify({"error": "No output from subprocess"}), 500

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if line.startswith("{") and line.endswith("}"):
                try:
                    json_output = json.loads(line)
                    return jsonify(json_output)
                except json.JSONDecodeError:
                    continue  

        return jsonify({"error": "No valid JSON output"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        result = subprocess.run(
            ["python", "train_model.py"],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        if result.stderr:
            print("❌ Training error:", result.stderr)

        if not result.stdout:
            return jsonify({"error": "No output from training process"}), 500

        return jsonify({"message": "Model training completed successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
