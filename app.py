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
    image_data = data.get("image")

    if not person_name or not image_data:
        return jsonify({"error": "Name and image are required"}), 400

    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        person_folder = os.path.join(dataset_path, person_name)
        os.makedirs(person_folder, exist_ok=True)

        count = len(os.listdir(person_folder))

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            if face_crop.size > 0:
                image_path = os.path.join(person_folder, f"{count}.jpg")
                cv2.imwrite(image_path, face_crop)
                count += 1

        return jsonify({"message": f"{count} face(s) saved for {person_name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
