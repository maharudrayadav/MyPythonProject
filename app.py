import json
from flask import Flask, jsonify, request
from PIL import Image
import subprocess
import os

app = Flask(__name__)

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    
    try:
        image = Image.open(image_file)  # Open image
        image = image.convert("RGB")  # Convert to RGB (if needed)
        
        # Convert image to NumPy array
        image_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Save grayscale image (for debugging)
        cv2.imwrite("gray_image.jpg", gray)

        return jsonify({"message": "Image processed successfully"}), 200

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
