import json
import subprocess
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import capture_faces  # ✅ Import capture_faces.py directly

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://cloud-app-dlme.onrender.com", "http://localhost:3000"]}})

@app.route("/capture_faces", methods=["POST"])
def capture_faces_endpoint():
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "Name is required"}), 400

    person_name = data["name"]
    capture_faces.capture_faces_function(person_name)  # ✅ Run inside Flask process
    return jsonify({"message": f"Capturing started for {person_name}"}), 202

@app.route("/recognize_faces", methods=["POST"])
def recognize_faces():
    try:
        result = subprocess.run(
            [sys.executable, "recognize_faces.py"],  # ✅ Use sys.executable
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
            [sys.executable, "train_model.py"],  # ✅ Use sys.executable
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
    port = int(os.environ.get("PORT", 5000))  # ✅ Use dynamic port for Render
    app.run(host="0.0.0.0", port=port, debug=False)
