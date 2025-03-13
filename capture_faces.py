import os
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

def capture_faces_function(user_name, file):
    dataset_path = "dataset"
    person_folder = os.path.join(dataset_path, user_name)
    os.makedirs(person_folder, exist_ok=True)

    image_path = os.path.join(person_folder, file.filename)
    file.save(image_path)  # ✅ Save the uploaded image
    print(f"✅ Image saved: {image_path}")

    return {"message": "Image captured successfully", "path": image_path}

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing file or name"}), 400

    file = request.files["image"]
    user_name = request.form["name"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    result = capture_faces_function(user_name, file)
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
